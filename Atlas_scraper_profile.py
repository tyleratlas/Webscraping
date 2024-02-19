#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import csv
import sys
from typing import Any, Dict, List
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_profile_data(ticker_symbols: List[str]) -> List[Dict[str, Any]]:
    all_stock_data = []
    
    for ticker_symbol in ticker_symbols:
        print('Getting stock data of', ticker_symbol)

        # Set user agent to avoid detection as a scraper
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0'}

        # Construct URL for the given ticker_symbol
        url = f'https://finance.yahoo.com/quote/{ticker_symbol}/profile'

        # Make a request to the URL
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')

        # Extract stock data from the HTML
        stock_data = {
            # scraping the stock data
            'stock_name': soup.find('div', {'class':'D(ib) Mt(-5px) Maw(38%)--tab768 Maw(38%) Mend(10px) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'}).find_all('div')[0].text.strip(),
            
            'address': soup.find('div', {'class':'Mb(25px)'}).find_all('p')[0].text.strip(),
        
            'executive_1': soup.find('table', {'class':'W(100%)'}).find_all('td')[0].text.strip(),
            'executive_2': soup.find('table', {'class':'W(100%)'}).find_all('td')[5].text.strip(),
            'executive_3': soup.find('table', {'class':'W(100%)'}).find_all('td')[10].text.strip(),
            'executive_4': soup.find('table', {'class':'W(100%)'}).find_all('td')[15].text.strip(),
            'executive_5': soup.find('table', {'class':'W(100%)'}).find_all('td')[20].text.strip(),
            'executive_6': soup.find('table', {'class':'W(100%)'}).find_all('td')[25].text.strip(),
            'executive_7': soup.find('table', {'class':'W(100%)'}).find_all('td')[30].text.strip(),
            'executive_8': soup.find('table', {'class':'W(100%)'}).find_all('td')[35].text.strip(),
            'executive_9': soup.find('table', {'class':'W(100%)'}).find_all('td')[40].text.strip(),
            'executive_10': soup.find('table', {'class':'W(100%)'}).find_all('td')[45].text.strip(),
            
            'description': soup.find('p', {'class': 'Mt(15px) Lh(1.6)'}).text.strip(),
        }            
        all_stock_data.append(stock_data)
        
    return all_stock_data

# Define the ticker symbols
ticker_symbols = ['TSLA', 'AMZN', 'AAPL', 'META', 'NFLX', 'GOOG']

# Get stock data for each ticker symbol
stock_data = get_profile_data(ticker_symbols)

# Writing stock data to a JSON file
with open('Atlas_stock_profile_data.json', 'w', encoding='utf-8') as f:
    json.dump(stock_data, f)

# Writing stock data to a CSV file with aligned values
CSV_FILE_PATH = 'Atlas_stock_profile_data.csv'
with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = stock_data[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(stock_data)

# Writing stock data to an Excel file
EXCEL_FILE_PATH = 'Atlas_stock_profile_data.xlsx'
df = pd.DataFrame(stock_data)
df.to_excel(EXCEL_FILE_PATH, index=False)

print('Done!')


# In[ ]:




