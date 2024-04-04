import pandas as pd

# Load the second dataset
df2 = pd.read_csv('995,000_cleaned_dataset.csv', low_memory=False)

# Define chunk size
chunksize = 5000

# Create an empty list to hold the dataframes
dfs = []
total_chunks = 995000 // chunksize
print('total number of chunks to run: {}'.format(total_chunks))
i = 0
df2['id'] = df2['id'].astype('string')
# Load the first dataset in chunks
for chunk in pd.read_csv('995,000_rows.csv', chunksize=chunksize, low_memory=False):
    i+=1
    print('Processing chunk: {}'.format(i - 1))
    chunk = chunk.dropna(subset=['id', 'type', 'domain', 'content', 'title'])
    chunk = chunk.drop(columns=['keywords', 'source', 'tags', 'meta_description', 'summary'])
    chunk['authors'] = chunk['authors'].fillna('<none>')
    chunk['meta_keywords'] = chunk['meta_keywords'].fillna('<none>')
    chunk['id'] = chunk['id'].astype('string')
    merged_chunk = pd.merge(df2, chunk.drop(columns='content'), on='id', how='inner')
    # Append the merged chunk to the list
    dfs.append(merged_chunk)

# Concatenate all dataframes in the list
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged dataframe to a csv file
merged_df.to_csv('merged_dataset.csv', index=False)