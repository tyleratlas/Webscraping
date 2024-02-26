#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import necessary libraries
import pandas as pd
import numpy as np
import json
import csv
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

# Set up End and Start times for data grab
tech_list = ['AAPL', 'GOOG', 'TSLA']

end = datetime.now()
start = datetime(end.year - 25, end.month, end.day)

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)
    
company_list = [AAPL, GOOG, TSLA]
company_name = ["APPLE", "GOOGLE", "TESLA"]

# Iterate through each company to write data to JSON, CSV, and XLSX files
for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
    # Writing stock data to a JSON file
    json_file_path = f"{com_name}_historical_data.json"
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(company.to_json(orient='index'), f)
    
    # Writing stock data to a CSV file
    csv_file_path = f"{com_name}_historical_data.csv"
    company.to_csv(csv_file_path)
    
    # Writing stock data to an Excel file
    excel_file_path = f"{com_name}_historical_data.xlsx"
    company.to_excel(excel_file_path, index=True)
    
df = pd.concat(company_list, axis=0)

print('Done!')


# In[ ]:




