# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:25:24 2024

@author: User
"""

import pandas as pd
from bs4 import BeautifulSoup
import requests

df = pd.DataFrame({
    "topic": ["Plant Care"],
    "subtopic": ["Plant nutrients"],
    "content": ["To improve photosynthesis, nutrient levels, and growth while identifying system limitations."],
    "tags": ["gas exchange, light response curves"],
    "source": ["https://dx.doi.org/10.3390/horticulturae9030291"]
})

# read html
html_path = df["source"].values[0]
print('The path for the html is: ', html_path)


# Extract text from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return text

# Use the function to extract text
text = extract_text_from_url(html_path)
print(text)
