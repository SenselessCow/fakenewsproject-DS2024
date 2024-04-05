# fakenewsproject-DS2024
Grundlæggende Data Science 2024 gruppeopgave

## Installation

To run this project, you need to install the required packages. You can do this by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Dataset Exploration

To explore the `995,000_rows.csv` dataset, you need to run the `explore_dataset.py` script. This script requires the `995,000_rows.csv` file as input.

The script outputs information about the dataset both before and after processing to remove and modify empty content. The output is written to two text files: `explore_995000_before.txt` and `explore_995000_after.txt`.

To run the script, use the following command:

```bash
python explore_dataset.py
```

## Data Cleaning

To clean the dataset, you need to run the `clean_dataset.py` script. This script requires the `995,000_rows.csv` file as input.

The script processes the data in chunks to accommodate for memory constraints. You can adjust the `chunksize` variable in the script depending on your system's memory.

The script outputs a cleaned dataset in a file named `995,000_cleaned_dataset.csv`. This file contains two columns: `id` and `content`. The `content` column contains the cleaned text data.

Later merge.py scrip was added which needs to be run after to merge type etc into the df aswell.

To run the script, use the following command:

```bash
python clean_dataset.py
python merge.py
```

## Model utilization

To utilize the model, you need to run the `script.py` script, which requires the `merged_dataset.csv` is available.

The script first extracts 10k random lines from the dataset for testing. Thus, results might not be entirely consistent, as a new dataset is created each time.

To run it, simply run the script.py, like so:

```bash
python script.py
```
