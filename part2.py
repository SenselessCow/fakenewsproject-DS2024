import pandas as pd
from collections import Counter
import ast
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Part 2
print('Part 2')
filename = "data/merged_dataset.csv"
filename2 = "data/995,000_rows.csv"
chunksize = 10000
total_chunks = 245286 // chunksize
print(total_chunks)

# Initialize the Counters
word_counts_pre_processed = Counter()
word_counts_post_processed = Counter()
type_counts = Counter()
i = 0
# Get raw data and for processing
for data_chunk in pd.read_csv(filename, chunksize=chunksize, low_memory=False):
      i+=1
      print(i)
      # print (data_chunk['content'].head(5))
      data_chunk['content'] = data_chunk['content'].apply(ast.literal_eval)
      data_chunk['content'].apply(word_counts_post_processed.update)
      
most_common_words = word_counts_post_processed.most_common(1000)
word_freq = dict(most_common_words)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide the axes
plt.title('1000 Most Frequent Words')
plt.savefig('most_frequent_words_wordcloud.png')

# most_common_words = word_counts_post_processed.most_common(1000)
# words = [word for word, count in most_common_words]
# counts = [count for word, count in most_common_words]

# plt.figure(figsize=(10, 15))
# plt.barh(words[:50], counts[:50])  # Display only the top 50 words for clarity
# plt.gca().invert_yaxis()
# plt.xscale('log')  # Set the x-axis to logarithmic scale
# plt.xlabel('Count (log scale)')
# plt.ylabel('Word')
# plt.title('Top 50 out of 1000 Most Frequent Words')
# plt.savefig('most_frequent_words_log_scale.png')

# print(type_counts)
# Compute the reduction rate and size of the vocabulary
# unique_word_count_pre = len(word_counts_pre_processed)
# unique_word_count_post = len(word_counts_post_processed)
# reduction_rate = dataprocessing.compute_reduction_rate(unique_word_count_pre, unique_word_count_post)
# print('Unique words before text processing: {},'
#       ' after text processing: {}'
#        ' and the reduction rate is: {:.2f}%'.format(unique_word_count_pre, unique_word_count_post, reduction_rate))