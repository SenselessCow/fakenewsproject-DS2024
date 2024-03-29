import dataprocessing
import models

# Part 1
print('Part 1')
url = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"
filename = "news_sample.csv"
raw_data = dataprocessing.get_data(url, filename)
cleaned_data = raw_data.copy()

raw_data['content'] = raw_data['content'].apply(dataprocessing.tokenize)
cleaned_data['content'] = cleaned_data['content'].apply(dataprocessing.text_preprocessing)

# Get the vocabulary of the raw data vs the processed data
total_word_count, unique_word_count, word_counts = dataprocessing.count_words(cleaned_data, 'content')
orig_total_word_count, orig_unique_word_count, orig_word_counts = dataprocessing.count_words(raw_data, 'content')
# Compute the reduction rate
reduction_rate = dataprocessing.compute_reduction_rate(orig_unique_word_count, unique_word_count)
print('Unique words before text processing: {},'
      ' after text processing: {}'
       ' and the reduction rate is: {:.2f}%'.format(orig_unique_word_count, unique_word_count, reduction_rate))


# Example of how to print the most common words
# print(word_counts.most_common(10))

# # Part 2
print('\nPart 2')
# filename = "995,000_rows.csv"
# # Get raw data and copies for processing
# raw_data_plus = dataprocessing.get_data(filename)
# cleaned_data_plus = raw_data_plus.copy()

# raw_data_plus = raw_data_plus['content'].apply(dataprocessing.tokenize)
# cleaned_data_plus['content'] = cleaned_data_plus['content'].apply(dataprocessing.text_preprocessing)

# # Get the vocabulary of the raw data vs the processed data
# total_word_count_plus, unique_word_count_plus, word_counts_plus = dataprocessing.count_words(cleaned_data_plus, 'content')
# orig_total_word_count_plus, orig_unique_word_count_plus, orig_word_counts_plus = dataprocessing.count_words(raw_data_plus, 'content')
# # Compute the reduction rate
# reduction_rate_plus = dataprocessing.compute_reduction_rate(orig_unique_word_count_plus, unique_word_count_plus)
# print('Unique words before text processing: {},'
#       ' after text processing: {}'
#        ' and the reduction rate is: {:.2f}%'.format(orig_unique_word_count_plus, unique_word_count_plus, reduction_rate_plus))

print("Fakenews modelling and predictions:\n")
models.linear_model1(cleaned_data)
# models.linear_model1(cleaned_data_plus)
models.logistic_model2(cleaned_data)
# models.logistic_model2(cleaned_data_plus)