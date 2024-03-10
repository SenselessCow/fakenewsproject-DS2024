import requests
from bs4 import BeautifulSoup
import json
import time

def soupify_url(url):
    response = requests.get(url)
    contents = response.text
    return BeautifulSoup(contents, 'html.parser')

def extract_article_title(soup_element):
    # Find and store the headline, published date and author
    text = soup_element.find_all('p', class_="ssrcss-1q0x1qg-Paragraph e1jhz7w10")
    full_text = ' '.join(paragraph.text for paragraph in text)
    headline = soup_element.find(id='main-heading')
    published_date = soup_element.find(attrs={'data-testid': 'timestamp'}) 
    datetime_value = published_date['datetime'] if published_date else 'None'
    author = soup_element.find('div', class_="ssrcss-68pt20-Text-TextContributorName e8mq1e96")
    author_name = author.text if author else 'None'
    # Store the values in a dictionary
    article_dict = {}
    article_dict['text'] = full_text
    article_dict['headline'] = headline.text if headline else 'None'
    article_dict['published_date'] = datetime_value
    article_dict['author'] = author_name
    
    # For each article, find the <a> tag and extract the 'href'
    # links = [article.find('a').get('href') for article in articles if article.find('a')]
    
    return article_dict

base_url = "https://www.bbc.com"


# Open the text file for reading ('r' mode)
with open('article_links.txt', 'r') as file, open('article_info.json', 'w') as jsonfile:
    counter = 0

    # Loop through each line in the file
    for line in file:
        # Concatenate each line to the base_url
        url = base_url + line.strip()
        
        # Call the soupify_url function to get the BeautifulSoup object
        soup = soupify_url(url)
        
        # Call the extract_article_title function to extract the article information
        article_info = extract_article_title(soup)
        
        # Write the article information to the CSV file
        json.dump(article_info, jsonfile)

        # add a delay to avoid being blocked
        time.sleep(1)