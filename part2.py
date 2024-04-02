import dataprocessing
import pandas as pd
from collections import Counter

# Part 2
print('Part 2')
filename = "995,000_rows.csv"
chunksize = 1000
total_chunks = 995000 // chunksize
print(total_chunks)
# Initialize the word counts
word_counts_pre_processed = Counter()
word_counts_post_processed = Counter()
i = 0
# Get raw data and for processing
for raw_data_chunk in pd.read_csv(filename, chunksize=chunksize, low_memory=False):
      # raw_data_chunk = dataprocessing.get_data(chunk)
      i+=1
      print(i)
      # Drop empty rows and fill in empty authors and meta_keywords with <none>
      raw_data_chunk = raw_data_chunk.dropna(subset=['id', 'type', 'domain', 'content', 'title'])
      raw_data_chunk = raw_data_chunk.drop(columns=['keywords', 'source', 'tags', 'meta_description', 'summary'])
      raw_data_chunk['authors'] = raw_data_chunk['authors'].fillna('<none>')
      raw_data_chunk['meta_keywords'] = raw_data_chunk['meta_keywords'].fillna('<none>')
      # Convert the data types to string, small memory optimization
      raw_data_chunk = raw_data_chunk.astype('string')

      # Work on an copy that gets cleaned
      cleaned_data_chunk = raw_data_chunk.copy()
      cleaned_data_chunk['content'] = cleaned_data_chunk['content'].apply(dataprocessing.text_preprocessing)

      # Tokenize the content
      raw_data_chunk['content'] = raw_data_chunk['content'].apply(dataprocessing.tokenize)

      # Get word counts and reduction rate
      word_counts_pre_processed = dataprocessing.count_words_extended(raw_data_chunk, 'content', word_counts_pre_processed)
      word_counts_post_processed = dataprocessing.count_words_extended(cleaned_data_chunk, 'content', word_counts_post_processed)

# Compute the reduction rate and size of the vocabulary
unique_word_count_pre = len(word_counts_pre_processed)
unique_word_count_post = len(word_counts_post_processed)
reduction_rate = dataprocessing.compute_reduction_rate(unique_word_count_pre, unique_word_count_post)
print('Unique words before text processing: {},'
      ' after text processing: {}'
       ' and the reduction rate is: {:.2f}%'.format(unique_word_count_pre, unique_word_count_post, reduction_rate))