#!/usr/bin/env python
# coding: utf-8

# In[125]:


# Import necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf

# For time stamps
from datetime import datetime

# Set up End and Start times for data grab
tech_list = ['AAPL', 'GOOG', 'TSLA']

end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)
    
company_list = [AAPL, GOOG, TSLA]
company_name = ["APPLE", "GOOGLE", "TESLA"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)


# In[107]:


df.head(5)


# In[108]:


df.tail(5)


# In[109]:


AAPL.describe()


# In[22]:


GOOG.describe()


# In[23]:


TSLA.describe()


# In[111]:


plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['High'].plot()
    plt.ylabel('High')
    plt.xlabel(None)
    plt.title(f"High of {tech_list[i - 1]}")
    
plt.tight_layout()


# In[25]:


import matplotlib.pyplot as plt
# Plot the training set
AAPL["High"][:'2022'].plot(figsize=(16, 4), legend=True)
# Plot the test set
AAPL["High"]['2023':].plot(figsize=(16, 4), legend=True)
plt.legend(['Training set (Before 2023)', 'Test set (2023 and beyond)'])
plt.title('AAPL stock price')
plt.show()


# In[26]:


# Plot the training set
GOOG["High"][:'2022'].plot(figsize=(16, 4), legend=True)
# Plot the test set
GOOG["High"]['2023':].plot(figsize=(16, 4), legend=True)
plt.legend(['Training set (Before 2023)', 'Test set (2023 and beyond)'])
plt.title('GOOG stock price')
plt.show()


# In[27]:


# Plot the training set
TSLA["High"][:'2022'].plot(figsize=(16, 4), legend=True)
# Plot the test set
TSLA["High"]['2023':].plot(figsize=(16, 4), legend=True)
plt.legend(['Training set (Before 2023)', 'Test set (2023 and beyond)'])
plt.title('TSLA stock price')
plt.show()


# In[62]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming df is defined earlier in the code
training_set = df[:'2022'].iloc[:,1:2].values
test_set = df['2023':].iloc[:, 1:2].values

# Scaling the training set
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Convert the training data to PyTorch tensors
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = torch.tensor(X_train).unsqueeze(2).float()
y_train = torch.tensor(y_train).unsqueeze(1).float()


# In[51]:


# Define the LSTM model
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model, loss function, and optimizer
regressor = LSTMRegressor(input_size=1, hidden_size=100, num_layers=4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = regressor(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# In[75]:


# Pre-process the AAPL test data
dataset_total = pd.concat((AAPL["High"][:'2022'], AAPL["High"]['2023':]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs_scaled = sc.transform(inputs)
X_test = []
for i in range(60, len(inputs_scaled)):
    X_test.append(inputs_scaled[i - 60:i, 0])
X_test = np.array(X_test)
X_test = torch.tensor(X_test).unsqueeze(2).float()

# Predict the stock price
predicted_AAPL_price = regressor(X_test)
predicted_AAPL_price = sc.inverse_transform(predicted_AAPL_price.detach().numpy())


# In[97]:


def plot_prediction(test,prediction):
    plt.plot(test,color='red',label="Real AAPL Stock Price")
    plt.plot(prediction, color="blue",label="Predicted AAPL Stock price")
    plt.title("AAPL Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("AAPL Stock Price")
    plt.legend()
    plt.show()
# now we'll use this function to visualize our test and predicted data

plot_prediction(AAPL['2022':'2024'].iloc[:, 1:2].values,predicted_AAPL_price)


# In[82]:


# Pre-process the GOOG test data
dataset_total = pd.concat((GOOG["High"][:'2022'], GOOG["High"]['2023':]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs_scaled = sc.transform(inputs)
X_test = []
for i in range(60, len(inputs_scaled)):
    X_test.append(inputs_scaled[i - 60:i, 0])
X_test = np.array(X_test)
X_test = torch.tensor(X_test).unsqueeze(2).float()

# Predict the stock price
predicted_GOOG_price = regressor(X_test)
predicted_GOOG_price = sc.inverse_transform(predicted_GOOG_price.detach().numpy())


# In[120]:


def plot_prediction(test,prediction):
    plt.plot(test,color='red',label="Real GOOG Stock Price")
    plt.plot(prediction, color="blue",label="Predicted GOOG Stock price")
    plt.title("GOOG Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("GOOG Stock Price")
    plt.legend()
    plt.show()
# now we'll use this function to visualize our test and predicted data

plot_prediction(GOOG['2021':'2024'].iloc[:, 1:2].values,predicted_GOOG_price)


# In[84]:


# Pre-process the TSLA test data
dataset_total = pd.concat((TSLA["High"][:'2022'], TSLA["High"]['2023':]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs_scaled = sc.transform(inputs)
X_test = []
for i in range(60, len(inputs_scaled)):
    X_test.append(inputs_scaled[i - 60:i, 0])
X_test = np.array(X_test)
X_test = torch.tensor(X_test).unsqueeze(2).float()

# Predict the stock price
predicted_TSLA_price = regressor(X_test)
predicted_TSLA_price = sc.inverse_transform(predicted_TSLA_price.detach().numpy())


# In[103]:


def plot_prediction(test,prediction):
    plt.plot(test,color='red',label="Real TSLA Stock Price")
    plt.plot(prediction, color="blue",label="Predicted TSLA Stock price")
    plt.title("TSLA Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("TSLA Stock Price")
    plt.legend()
    plt.show()
# now we'll use this function to visualize our test and predicted data

plot_prediction(TSLA['2021':'2024'].iloc[:, 1:2].values,predicted_TSLA_price)


# In[ ]:




