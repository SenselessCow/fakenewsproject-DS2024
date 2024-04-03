import dataprocessing
import pandas as pd

# Clean data and save to a csv
filename = "995,000_rows.csv"
chunksize = 10000
total_chunks = 995000 // chunksize
print('total number of chunks to run: {}'.format(total_chunks))
i = 0
# Get raw data and for processing
for raw_data_chunk in pd.read_csv(filename, chunksize=chunksize, low_memory=False):
      # raw_data_chunk = dataprocessing.get_data(chunk)
      i+=1
      print('Processing chunk: {}'.format(i - 1))
      # Drop empty rows and fill in empty authors and meta_keywords with <none>
      raw_data_chunk = raw_data_chunk.dropna(subset=['id', 'type', 'domain', 'content', 'title'])
      raw_data_chunk = raw_data_chunk.drop(columns=['keywords', 'source', 'tags', 'meta_description', 'summary'])
      raw_data_chunk['authors'] = raw_data_chunk['authors'].fillna('<none>')
      raw_data_chunk['meta_keywords'] = raw_data_chunk['meta_keywords'].fillna('<none>')
      # Convert the data types to string, small memory optimization
      raw_data_chunk = raw_data_chunk.astype('string')

      # Create cleaned_data_chunk which only contains the id and content column
      cleaned_data_chunk = raw_data_chunk[['id', 'content']].copy()
      cleaned_data_chunk['content'] = cleaned_data_chunk['content'].apply(dataprocessing.text_preprocessing)

      # Append the cleaned data to a the same csv file
      cleaned_data_chunk.to_csv('995,000_cleaned_dataset.csv', mode='a', header=(i == 1), index=False)
