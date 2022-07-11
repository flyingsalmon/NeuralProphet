#!/usr/bin/env python
# coding: utf-8

#  A Neural Network based Time-Series model, inspired by Facebook Prophet and AR-Net, built on PyTorch
#     Documentation: https://neuralprophet.com/html/index.html

#  Following code is an export from my Jupyer notebook code.

# In[100]:
get_ipython().system('pip install neuralprophet')

# NOTE: pickle is included in standard python lib by default from Pyhon 3.x

# In[101]:
import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
import pickle # for saving to disk


# In[102]:
DATASET = 'seattleWeather_1948-2017.xlsx'

# In[103]:
df=pd.read_excel(DATASET)
print(df.info)


# Check what data types are each column...

# In[104]:


df.dtypes
# We see DATE column is a datetime object which is exactly what we need.
# Neural Prophet only takes a datetime object type.

# In[105]:

plt.plot(df['DATE'], df['TMAX'])

# In[106]:
df['YR'] = df['DATE'].apply(lambda x: x.year) # grab only the year portion via lambda from DATE col.
df = df[df['YR'] >=1967]
plt.plot(df['DATE'], df['TMAX'])
plt.show()
# Now we see the max temp data for 50 years To-date (2017).


# In[107]:
df.head() # print the first 5 rows of our df and we see 'YR' as a new column...

# In[108]:
df.tail() # print the last 5 rows of our df and we see 'YR' as a new column...


# Neural Prophet ONLY supports 2 columns and 2 columns only! The first column is ds (for dates) and the second is 
# y (value that we want to predict...e.g. maximum temp)

# In[109]:
df2 = df[['DATE', 'TMAX']] 
df2.dropna(inplace=True) # drop any missing rows of data from the df2 in place
df2.columns = ['ds', 'y'] # Now rename the two columns to ds and y as Prophet requires
df2.head() 


# In[110]:
#Next, we need to train the Neural Prophet model...
model = NeuralProphet() # create a model
model.fit(df2, freq='D') # train the model by fit()
# 'D' means use daily data from dataset as our data is daily in the dataset
# MAE: Mean Absolute Error is 5.1 or 5.1 points of error (+/-).

# In[119]:
# It's time to predict...

# In[112]:
future = model.make_future_dataframe(df2, periods=180) 
# periods specifies how many periods we're predicting, e.g. n days since our freq='D' (daily)
# 180 periods == 6 months daily forecast


# In[113]:
forecast = model.predict(future)
forecast.head() # print the first 5 days or predicted temps


# NOTE: yhat1 ("y-hat one") column contains the forecast temperatures.

# In[114]:
forecast.tail() # print the last 5 days or predicted temps


# To see only the ds (date) and yhat (predicted TMAX temp) columns, just filter the print() as below:
# forecast[['ds', 'yhat1']] but don't do this before plotting or it may generate keyerror in plotting!
# We can plot the forecast temp values (in y axis and dates in x axis)...

# In[115]:
plot1 = model.plot(forecast)

# The predicted data can also be shown like a visualization broken down by different components
# such as overl trend, seasonality by year and week easily by:
#     model.plot_components()

# In[117]:

plot2 = model.plot_components(forecast)

# NOTE: the y-axis in Trend chart shows the max temp, but in the Seasonality charts above y-axis is showing difference/fluctuations
#     of temps.

# OPTIONAL: To save this model to a file (persist), we use pickle library..

# In[118]:

with open('WeatherPredictionNPSaved.pkl', "wb") as f:
          pickle.dump(model, f)
        
# The output binary file will be created and written as binary in the current directory of this notebook...
# To access it, File->Open (which opens http://localhost:8888/tree) and we'll see the WeatherPredictionNPSaved.pkl
# file there.

# To load the model in the future, use the following snippet:
# 
#     with open('WeatherPredictionNPSaved.pkl', "rb") as f:
#         model = pickle.load(f)
# 
# Then after loading a saved model, we can predict for another timeframe by running the following code block:
# And change the periods for example:
# 
# future = model.make_future_dataframe(df2, periods=100) # 100 days ahead for example
# forecast = model.predict(future)
# forecast.head()
# 
# Then we can replot the new results by:
# plot1 = model.plot(forecast)

# the end
