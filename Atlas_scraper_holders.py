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

def get_holders_data(ticker_symbols: List[str]) -> List[Dict[str, Any]]:
    all_stock_data = []
    
    for ticker_symbol in ticker_symbols:
        print('Getting stock data of', ticker_symbol)

        # Set user agent to avoid detection as a scraper
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0'}

        # Construct URL for the given ticker_symbol
        url = f'https://finance.yahoo.com/quote/{ticker_symbol}/holders'

        # Make a request to the URL
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')

        # Extract stock data from the HTML
        stock_data = {
            # scraping the stock data
            'stock_name': soup.find('div', {'class':'D(ib) Mt(-5px) Maw(38%)--tab768 Maw(38%) Mend(10px) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'}).find_all('div')[0].text.strip(),
            
            'major_holders_1': soup.find('div', {'class': 'W(100%) Mb(20px)'}).find_all('tr')[0].text.strip(),
            'major_holders_2': soup.find('div', {'class': 'W(100%) Mb(20px)'}).find_all('tr')[1].text.strip(),
            'major_holders_3': soup.find('div', {'class': 'W(100%) Mb(20px)'}).find_all('tr')[2].text.strip(),
            'major_holders_4': soup.find('div', {'class': 'W(100%) Mb(20px)'}).find_all('tr')[3].text.strip(),
            
            'top_institutional_holders_1': soup.find('div', {'class': 'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[0].text.strip(),
            'top_institutional_holders_2': soup.find('div', {'class':'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[5].text.strip(),
            'top_institutional_holders_3': soup.find('div', {'class':'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[10].text.strip(),
            'top_institutional_holders_4': soup.find('div', {'class':'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[15].text.strip(),
            'top_institutional_holders_5': soup.find('div', {'class':'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[20].text.strip(),
            'top_institutional_holders_6': soup.find('div', {'class':'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[25].text.strip(),
            'top_institutional_holders_7': soup.find('div', {'class':'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[30].text.strip(),
            'top_institutional_holders_8': soup.find('div', {'class':'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[35].text.strip(),
            'top_institutional_holders_9': soup.find('div', {'class':'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[40].text.strip(),
            'top_institutional_holders_10': soup.find('div', {'class':'Mt(25px) Ovx(a) W(100%)'}).find_all('td')[45].text.strip(),
        }     
        all_stock_data.append(stock_data)
        
    return all_stock_data

# Define the ticker symbols
ticker_symbols = ['TSLA', 'AMZN', 'AAPL', 'META', 'NFLX', 'GOOG']

# Get stock data for each ticker symbol
stock_data = get_holders_data(ticker_symbols)

# Writing stock data to a JSON file
with open('Atlas_stock_holders_data.json', 'w', encoding='utf-8') as f:
    json.dump(stock_data, f)

# Writing stock data to a CSV file with aligned values
CSV_FILE_PATH = 'Atlas_stock_holders_data.csv'
with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = stock_data[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(stock_data)

# Writing stock data to an Excel file
EXCEL_FILE_PATH = 'Atlas_stock_holders_data.xlsx'
df = pd.DataFrame(stock_data)
df.to_excel(EXCEL_FILE_PATH, index=False)

print('Done!')


# In[ ]:




