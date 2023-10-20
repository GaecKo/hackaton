#%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


#1. split the dataset into training set and validation set

# ===== Load CSV & convert dates ===== #

df = pd.read_csv('Radiation.csv')
df['YEAR'] = pd.to_datetime(df['DATE'], format='%Y%m%d').dt.year

# ===== Divide datasets ===== #

t_set = df[(df['YEAR'] >= 1977) & (df['YEAR'] <= 2010)]   #1977-2010
training_set = t_set['Caen']
v_set = df[(df['YEAR'] >= 2011) & (df['YEAR'] <= 2019)] #2011-2019
validation_set = v_set['Caen']


#2. estimate this model with statsmodel on the training set

model = AutoReg(training_set, lags=10)
res = model.fit()
print(res.summary())

#3. judge quality

#4. compute the Mean Absolute error (MAE) on the training set