import dataprocessing
from collections import Counter
import re

raw_data = dataprocessing.get_data()
tokenized_raw_data = raw_data.copy()
cleaned_data = raw_data.copy()

tokenized_raw_data['content'] = tokenized_raw_data['content'].apply(dataprocessing.tokenize)
cleaned_data['content'] = cleaned_data['content'].apply(dataprocessing.text_preprocessing)

# Get the vocabulary of the raw data vs the processed data
total_word_count, unique_word_count, word_counts = dataprocessing.count_words(cleaned_data, 'content')
orig_total_word_count, orig_unique_word_count, orig_word_counts = dataprocessing.count_words(tokenized_raw_data, 'content')
# Compute the reduction rate
reduction_rate = dataprocessing.compute_reduction_rate(orig_unique_word_count, unique_word_count)
print('Unique words before text processing: {},'
      ' after text processing: {}'
       ' and the reduction rate is: {:.2f}%'.format(orig_unique_word_count, unique_word_count, reduction_rate))

# print(word_counts.most_common(10))
# print(orig_word_counts.most_common(10))
# unique_word_count = len(set(results))
# print('Unique words after simple cleaning: {}'.format(unique_word_count))