import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from cleantext import clean
import datefinder
import os
import pandas as pd 

def get_data():
    if os.path.isfile('news_sample.csv'):
        raw_data = pd.read_csv('news_sample.csv', index_col = 0)
    else:
        raw_data = pd.read_csv("https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv", index_col = 0)
    return raw_data

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

check_nltk_package('punkt')
check_nltk_package('stopwords')

def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+|<\w+>|[^<\w\s]+')
    tokens = tokenizer.tokenize(text)
    return tokens

def remove_stopwords(x):
    stop_words = set(stopwords.words('english'))
    return [w for w in x if not w.lower() in stop_words]

def stem_data(x):
    ps = PorterStemmer()
    return [ps.stem(w) for w in x]

def text_preprocessing(x):
    x = clean_data(x)
    x = replace_dates_with_token(x)
    x = tokenize(x)
    x = remove_stopwords(x)
    x = stem_data(x)
    return x