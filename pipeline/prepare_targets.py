import pandas as pd
import numpy as np
from pathlib import Path
import boto3
import json

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATA_DIR_IBLI = "s3://amazon-sagemaker-575108933641-us-east-1-c422b90ce861/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/local-uploads/1771807288486/IBLIData_CSV_PublicZipped"
EXPORT_DIR = "s3://amazon-sagemaker-575108933641-us-east-1-c422b90ce861/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/data/training/ibli"

# Load column mapping
s3 = boto3.client("s3")
bucket = "lmr-capstone-s3bucket"
key = "data/training/ibli/column_mapping.json"
response = s3.get_object(Bucket=bucket, Key=key)
col_map = json.loads(response["Body"].read())

# ------------------------------------------------------------------
# Load required datasets
# ------------------------------------------------------------------
losses = pd.read_csv(f"{DATA_DIR_IBLI}/S6C Livestock Losses.csv") \
           .rename(columns=col_map["losses"])

locations = pd.read_csv(f"{DATA_DIR_IBLI}/HH_location_shifted.csv")

stock_df_init = pd.read_csv(f"{DATA_DIR_IBLI}/S6A Livestock Stock.csv")\
    .rename(columns={'s6q1':'total_animals_herded',
                     's6q2':'herded_animals_owned',
                     's6q3':'herded_animals_adult',
                     's6q4':'herded_animals_always_home',
                     's6q5':'herded_animals_lactating'})\
    .drop(columns=['comment', 's6q6', 's6q7', 's6q66', 's6q67'])

# ------------------------------------------------------------------
# Clean and forward/backfill GPS coordinates (record-on-change logic)
# ------------------------------------------------------------------
locations = locations.sort_values(["hhid", "round"])
locations["gps_latitude"] = (
    locations.groupby("hhid")["gps_latitude"].ffill().bfill()
)
locations["gps_longitude"] = (
    locations.groupby("hhid")["gps_longitude"].ffill().bfill()
)

# ------------------------------------------------------------------
# TLU conversion
# ------------------------------------------------------------------
TLU_RATES = {
    "Cattle": 1.0,
    "Camels": 1.3,
    "Goats": 0.1,
    "Sheep": 0.1,
    "Goat/Sheep": 0.1,
    " Goat/Sheep": 0.1,
}

def get_tlu_rate(animal_type):
    if pd.isna(animal_type):
        return 0.1
    animal_type = str(animal_type).strip()
    if animal_type in TLU_RATES:
        return TLU_RATES[animal_type]
    animal_lower = animal_type.lower()
    if "cattle" in animal_lower:
        return 1.0
    elif "camel" in animal_lower:
        return 1.3
    else:
        return 0.1

losses["tlu_rate"] = losses["livestock_type_raw"].apply(get_tlu_rate)

if "quantity" in losses.columns:
    losses["quantity"] = pd.to_numeric(
        losses["quantity"], errors="coerce"
    ).fillna(1)
else:
    losses["quantity"] = 1

losses["tlu_loss"] = losses["tlu_rate"] * losses["quantity"]

# ------------------------------------------------------------------
# Merge with coordinates and filter valid spatial records
# ------------------------------------------------------------------
coords = locations[["hhid", "gps_longitude", "gps_latitude"]] \
            .drop_duplicates(subset=["hhid"])

valid_spatial = losses.merge(coords, on="hhid", how="left")

valid_spatial = valid_spatial.dropna(
    subset=["gps_longitude", "gps_latitude", "tlu_loss"]
)

valid_spatial["tlu_loss"] = pd.to_numeric(
    valid_spatial["tlu_loss"], errors="coerce"
).fillna(0)

# .a and .b correspond to 'I don't know' and 'refuse to answer' -- drop rows where these are the year/month
valid_spatial = valid_spatial.loc[~((valid_spatial.year=='.a')|
                     (valid_spatial.month=='.a')|
                     (valid_spatial.year=='.b')|
                     (valid_spatial.month=='.b'))]

# ------------------------------------------------------------------
# Export spatial dataset -- prepared using Matt's logic
# ------------------------------------------------------------------
path = f"{EXPORT_DIR}/livestock_losses_spatial_pipeline.csv"
valid_spatial.to_csv(path, index=False)
print(f'Exported spatial dataset: {path}')

# ------------------------------------------------------------------
# Prepare training data
# ------------------------------------------------------------------
valid_spatial['hhid'] = valid_spatial['hhid'].astype(int)
valid_spatial['year'] = valid_spatial['year'].astype(int)
valid_spatial['month'] = valid_spatial['month'].astype(int)

# ------------------------------------------------------------------
# Calculate TLU loss as a proportion of livestock stock
# ------------------------------------------------------------------

def compute_tlu_vectorized(df, animal_type_col, numeric_col):
    """
    Vectorized computation of TLU-scaled livestock values.
    """
    animal = (
        df[animal_type_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    rates_lookup = {k.lower().strip(): v for k, v in TLU_RATES.items()}
    # Start with exact matches
    rates = animal.map(rates_lookup)
    
    # Handle fuzzy matches where exact match failed
    rates = rates.fillna(
        animal.str.contains("cattle", na=False).astype(float) * 1.0
    )
    rates = rates.mask(
        animal.str.contains("camel", na=False),
        1.3
    )
    # Anything still missing → default shoat rate
    rates = rates.fillna(0.1)

    # handle non-numeric values, if non-numeric becomes NaN
    counts = pd.to_numeric(df[numeric_col], errors="coerce")
    
    return counts * rates

# Get TLU for relevant columns of stock
for col in ['total_animals_herded','herded_animals_owned', 'herded_animals_adult', 'herded_animals_always_home','herded_animals_lactating']:
    stock_df_init[col] = pd.to_numeric(stock_df_init[col], errors="coerce") # remove .a/.b
    stock_df_init[f"{col}_tlu_rate"] = compute_tlu_vectorized(stock_df_init,'animaltype',col)

stock_df = stock_df_init.groupby(['hhid','round'])['total_animals_herded_tlu_rate'].sum().reset_index()

# Create % of herd size target
losses_merged = valid_spatial.merge(stock_df, on=['hhid', 'round'])
losses_merged['tlu_loss_ratio'] = losses_merged['tlu_loss']/losses_merged['total_animals_herded_tlu_rate']

# ------------------------------------------------------------------
# Create monthly TLU loss dataset
# ------------------------------------------------------------------
# Loss data only has information when there's a loss. We need to do some manipulation to create a monthly HH dataset

# Round coordinates to avoid floating point precision issues
losses_merged['gps_latitude_rounded'] = losses_merged['gps_latitude'].round(6)
losses_merged['gps_longitude_rounded'] = losses_merged['gps_longitude'].round(6)

# Assume date is month-end
losses_merged['date'] = pd.to_datetime(losses_merged[['year', 'month']].assign(day=1)) + pd.offsets.MonthEnd(0)

# Continuous observation periods for each household-location
def get_continuous_periods(group):
    """Identify continuous date ranges, breaking at large gaps"""
    group = group.sort_values('date')
    group['date_diff'] = group['date'].diff()
    
    # Define a gap threshold
    # Allow for 2 years of 0-fill, this will naturally break at the 2016-2018 gap (>1000 days)
    group['new_period'] = (group['date_diff'] > pd.Timedelta(days=730)) | group['date_diff'].isna()
    group['period_id'] = group['new_period'].cumsum()
    
    return group

# Apply the function to identify periods
losses_with_periods = losses_merged.groupby(
    ['hhid', 'gps_latitude_rounded', 'gps_longitude_rounded'],
    group_keys=False
).apply(get_continuous_periods).reset_index(drop=True)

# agg losses by household, location, year, and month
monthly_losses = losses_with_periods.groupby([
    'hhid', 
    'gps_latitude_rounded', 
    'gps_longitude_rounded',
    'year', 
    'month',
    'period_id'
]).agg({
    'tlu_loss': 'sum',
    'tlu_loss_ratio': 'mean',
    'date': 'first'
}).reset_index()

period_ranges = losses_with_periods.groupby([
    'hhid',
    'gps_latitude_rounded',
    'gps_longitude_rounded',
    'period_id'
]).agg({
    'date': ['min', 'max']
}).reset_index()

period_ranges.columns = ['hhid', 'gps_latitude_rounded', 'gps_longitude_rounded', 
                         'period_id', 'min_date', 'max_date']

complete_data = []

for _, row in period_ranges.iterrows():
    hhid = row['hhid']
    lat = row['gps_latitude_rounded']
    lon = row['gps_longitude_rounded']
    period_id = row['period_id']
    
    # Generate all months in this continuous period
    date_range = pd.date_range(
        start=row['min_date'],
        end=row['max_date'],
        freq='ME'  # Month end frequency
    )
    
    for date in date_range:
        complete_data.append({
            'hhid': hhid,
            'gps_latitude_rounded': lat,
            'gps_longitude_rounded': lon,
            'period_id': period_id,
            'year': date.year,
            'month': date.month,
            'date': date
        })

complete_df = pd.DataFrame(complete_data)

# Merge with aggregated losses
result = complete_df.merge(
    monthly_losses,
    on=['hhid', 'gps_latitude_rounded', 'gps_longitude_rounded', 
        'year', 'month', 'period_id'],
    how='left',
    suffixes=('', '_agg')
)

if 'date_agg' in result.columns:
    result = result.drop(columns=['date_agg'])

# Ind for whether data was actually observed vs zero-filled
result['data_observed'] = result['tlu_loss'].notna().astype(int)

# Fill tlu_loss with 0 only for months within continuous periods where no loss was recorded
result['tlu_loss'] = result['tlu_loss'].fillna(0)

# add season using Matt's logic from notebook 01
result['season'] = np.where((result['month']>=3)&(result['month']<=9),
                            'LRLD',
                            'SRSD')

# Clean up
result = result.rename(columns={
    'gps_latitude_rounded': 'gps_latitude',
    'gps_longitude_rounded': 'gps_longitude',
    'date':'ibli_date',
})

result['ibli_dekad'] = 3
result = result.sort_values(
    ['hhid', 'gps_latitude', 'gps_longitude', 'period_id', 'year', 'month']
).reset_index(drop=True)

result = result[[
    'hhid', 
    'gps_latitude', 
    'gps_longitude',
    'ibli_date',
    'year', 
    'month',
    'ibli_dekad',
    'tlu_loss',
    'tlu_loss_ratio',
    'data_observed',
    'season'
]]

result['tlu_loss_ratio'] = np.where((result['tlu_loss']==0) & result['tlu_loss_ratio'].isna(),
                                    0,
                                    result['tlu_loss_ratio'])

result.to_csv(f"{EXPORT_DIR}/target_data_pipeline.csv", index=False)

print('\n',"="*25, "Created target dataset","="*25)
print(f"Total rows: {len(result)}")
print(f"Unique households: {result['hhid'].nunique()}")
print(f"Unique household-location combinations: {result.groupby(['hhid', 'gps_latitude', 'gps_longitude']).ngroups}")
print(f"\nObserved vs zero-filled:")
print(result['data_observed'].value_counts())

print(f"\nYears present in result:")
print(result['year'].value_counts().sort_index())