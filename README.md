Atlas_scraper_holders.py - 
Scrapes the holder data: (https://finance.yahoo.com/quote/{ticker_symbol}/holders' - 
Creates three output files: 'Atlas_stock_holders_data.json', Atlas_stock_holders_data.csv', Atlas_stock_holders_data.xlsx'

Atlas_scraper_profile.py - 
Scrapes the profile data: (https://finance.yahoo.com/quote/{ticker_symbol}/profile' - 
Creates three output files: 'Atlas_stock_profile_data.json', Atlas_stock_profile_data.csv', Atlas_stock_profiles_data.xlsx'

Atlas_Download_Historical_Data.py - 
Scrapes the historical data from AAPL, GOOG, & TSLA
Creates three output files for each stock: .json, .csv, .xlsx

Atlas_Time_Series_Model.py -
Builds Long Short-Term Memory (LSTM) model using PyTorch to predict the future price of AAPL, GOOG, & TSLA stocks
Splits data into training set (2019-2022) and test set (2023-2024)
Creates visualizations showing the actual stock price vs predicted stock price over time for each stock
