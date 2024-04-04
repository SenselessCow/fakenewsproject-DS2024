import data_manipulation_scripts.dataprocessing as dataprocessing
import sys

# Explore the data
print('Part 2')
filename = "../data/995,000_rows.csv"
# # Get raw data and copies for processing
raw_data_plus = dataprocessing.get_data(filename)

# Open a text file to write the output
with open('../data/explore_995000_before.txt', 'w') as f:
    # Redirect the output to the text file
    sys.stdout = f

    # Explore the dataset
    print(raw_data_plus.columns)
    print('Exploring the dataset...')
    print(raw_data_plus.describe())
    print('getting dtypes...')
    print(raw_data_plus.dtypes)
    print('getting null values...')
    print(raw_data_plus.isnull().sum())
    print('getting unique values...')
    for col in raw_data_plus.select_dtypes(include=['object']).columns:
        print(raw_data_plus[col].value_counts())
    # print('getting correlation...')
    # print(raw_data_plus.corr())
    for col in raw_data_plus.columns:
        print('Column: {} and {}'.format(col, raw_data_plus[col].apply(type).unique()))

sys.stdout = sys.__stdout__

# Drop empty rows and duplicates
raw_data_plus = raw_data_plus.dropna(subset=['id', 'type', 'domain', 'content', 'title'])
raw_data_plus = raw_data_plus.drop(columns=['keywords', 'source', 'tags', 'meta_description', 'summary'])

# Open a text file to write the output
with open('../data/explore_995000_after.txt', 'w') as f:
    # Redirect the output to the text file
    sys.stdout = f

    # Explore the dataset
    print(raw_data_plus.columns)
    print('Exploring the dataset...')
    print(raw_data_plus.describe())
    print('getting dtypes...')
    print(raw_data_plus.dtypes)
    print('getting null values...')
    print(raw_data_plus.isnull().sum())
    print('getting unique values...')
    for col in raw_data_plus.select_dtypes(include=['object']).columns:
        print(raw_data_plus[col].value_counts())
    for col in raw_data_plus.columns:
        print('Column: {} and {}'.format(col, raw_data_plus[col].apply(type).unique()))
    # print('getting correlation...')
    # print(raw_data_plus.corr())

sys.stdout = sys.__stdout__
