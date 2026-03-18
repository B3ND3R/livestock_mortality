# Pipeline for Training LMR Model

## Prerequisites

### Data Directory Structure
Assumes Kenya survey (IBLI) data is organized in the following directory structure:
```
./data
└── IBLIData_CSV_PublicZipped
    ├── HH_location_shifted.csv
    ├── IBLI_sales.csv
    ├── S0A Household Identification information.csv
    ├── S0B Comments.csv
    ├── S1 Household Information.csv
    ├── S10 Herd Migration and Satellite Camps.csv
    ...
    └── S9B Other Assistance.csv
```

## Create Training Dataset
```sh
cd pipeline
pixi run prepare_targets.py
```
Creates `data/target_data_pipeline.csv`

