import pandas as pd
from statsmodels.tsa.api import acf, graphics
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from sklearn.metrics import mean_absolute_error as mae 

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
result = model.fit()

print(result.summary())

#3. judge quality

#4. compute the Mean Absolute error (MAE) on the training set

actual_data = training_set
calculated_data = result.predict()

# ===== Check for NaN values ===== #

actual_data_nan = actual_data.isnull().any()
calc_data_nan = calculated_data.isnull().any()

# ===== Remove the missing values and make the sets the same size ===== #

if actual_data_nan | calc_data_nan:
    missing_rows = actual_data.isnull() | calculated_data.isnull()
    actual_data = actual_data[~missing_rows]
    calculated_data = calculated_data[~missing_rows]

# ===== Calculate the MAE ===== #


MAE = mae(actual_data, calculated_data)

print("The Mean Absolute Error between predicted and real consumptions on the training set is: {} W/m²".format(MAE))