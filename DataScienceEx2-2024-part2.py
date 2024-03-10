import requests
from bs4 import BeautifulSoup

response = requests.get('https://www.bbc.com/news/world/europe')
contents = response.text

soup = BeautifulSoup(contents, 'html.parser')

def extract_article_links(soup_element):
    # Find all elements that are marked as articles
    articles = soup_element.find_all(attrs={'type': 'article'})
    
    # For each article, find the <a> tag and extract the 'href'
    links = [article.find('a').get('href') for article in articles if article.find('a')]
    
    return links

links = extract_article_links(soup)

# Write a function that extracts articles from all the pages
def extract_articles_from_all_pages(country, num_pages):
    # Create an empty list to store the links
    all_links = []
    
    # Loop through the pages
    for page in range(1, num_pages):
        # Construct the URL
        url = f'https://www.bbc.com/news/world/{country}?page={page}'
        
        # Make the request
        response = requests.get(url)
        
        # Extract the contents
        contents = response.text
        
        # Create a soup object
        soup = BeautifulSoup(contents, 'html.parser')
        
        # Extract the links
        links = extract_article_links(soup)
        
        # Add the links to the list
        all_links.extend(links)
    
    return all_links

all_links = []

regions_and_pages = [("africa", 25), ("asia",42), ("australia", 42), ("europe", 42), ("latin_america", 42), ("middle_east", 41)]
all_links.extend(extract_articles_from_all_pages("africa", 25))
for region, pages in regions_and_pages:
    all_links.extend(extract_articles_from_all_pages(region, pages))

# Open a file for writing (this will create the file if it doesn't exist)
with open('article_links.txt', 'w') as file:
    # Iterate over the list and write each element to the file
    for item in all_links:
        file.write(f"{item}\n")
