import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from cleantext import clean
import datefinder
import os
import pandas as pd 
from collections import Counter
import random

def get_data(filename, url=None):
    if os.path.isfile(filename):
        raw_data = pd.read_csv(filename, index_col = 0)
    if url is not None and not os.path.isfile(filename):
        raw_data = pd.read_csv(url, index_col = 0)
        raw_data.to_csv(filename)
    if url is None:
        print('Please provide a url or a filename')
    return raw_data

def extract_random_10k(filename):
    ran_num = 200000 - random.randint(1, 100)*1000
    print(ran_num)
    skip_rows=range(1, ran_num)

    data_subset = pd.read_csv(filename, skiprows=skip_rows, nrows=10000)

    file_name_without_extension = filename.split('.')[0]
    
    new_filename = f"10k_{file_name_without_extension}.csv"
    
    data_subset.to_csv(new_filename, index=False)

def clean_data(x):
    return clean(x,
        fix_unicode=False,               # fix various unicode errors
        to_ascii=False,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=True,               # replace all numbers with a special token
        no_digits=True,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUM>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
    )

def replace_dates_with_token(text, token="<DATE>"):
    matches = datefinder.find_dates(text)
    for match in matches:
        text = text.replace(str(match), token)
    return text

def check_nltk_package(package_name):
    try:
        nltk.data.find('tokenizers/' + package_name)
        print(f"{package_name} is installed")
    except LookupError:
        print(f"{package_name} is not installed. Downloading...")
        nltk.download(package_name)

# check_nltk_package('punkt')
check_nltk_package('stopwords')

def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+|<\w+>|[^<\w\s]+')
    tokens = tokenizer.tokenize(text)
    return tokens

def remove_stopwords(x):
    stop_words = set(stopwords.words('english'))
    return [w for w in x if not w.lower() in stop_words]

# Remove . / , : ; etc
def remove_punctuation(x):
    # Replace punctuations with the empty string if you want a count of it
    # return [re.sub(r'\W+', '', word) if len(word) == 1 else word for word in x]
    # Remove puntuations if it is 1 char long and not alphanumeric
    return [word for word in x if len(word) > 1 or word.isalnum()]

def stem_data(x):
    ps = PorterStemmer()
    return [ps.stem(w) for w in x]

def text_preprocessing(x):
    x = clean_data(x)
    x = replace_dates_with_token(x)
    x = tokenize(x)
    x = remove_stopwords(x)
    x = remove_punctuation(x)
    x = stem_data(x)
    return x

def count_words(df, column_name):
    # Create a Counter object for word counts
    word_counts = Counter()

    # Update word counts for each row
    for sublist in df[column_name]:
        word_counts.update(sublist)

    # Calculate total and unique word counts
    total_word_count = sum(word_counts.values())
    unique_word_count = len(word_counts)

    return total_word_count, unique_word_count, word_counts

def compute_reduction_rate(orig_unique_word_count, unique_word_count):
    return ((orig_unique_word_count - unique_word_count) / orig_unique_word_count) * 100
