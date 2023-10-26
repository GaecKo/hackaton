import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error as mae 

#1. split the dataset into training set and validation set

# ===== Load CSV & convert dates ===== #

df = pd.read_csv('Radiation.csv')
df['YEAR'] = pd.to_datetime(df['DATE'], format='%Y%m%d').dt.year

# ===== Divide datasets ===== #

t_set = df[(df['YEAR'] >= 1977) & (df['YEAR'] <= 2010)]   #1977-2010
training_set = t_set[['YEAR', 'Caen']]
v_set = df[(df['YEAR'] >= 2011) & (df['YEAR'] <= 2019)] #2011-2019
validation_set = v_set[['YEAR', 'Caen']]

#2. estimate this model with statsmodel on the training set

model = AutoReg(training_set['Caen'], lags=10)
result = model.fit()

print(result.summary())

#3. judge quality

actual_data = training_set['Caen']
calculated_data = result.predict()


plt.plot(training_set['YEAR'], actual_data/10, label="Measured Temp.", linestyle="-", color="blue")
plt.plot(training_set['YEAR'], calculated_data.values/10, label="Predicted Temp.", linestyle="--", color="red")

plt.title("Measured temperature and predicted temperature by AutoReg model")
plt.xlabel("Date")
plt.ylabel("Temperature (celsius)")
plt.legend()
plt.show()

#4. compute the Mean Absolute error (MAE) on the training set

# ===== Calculate the MAE ===== #

MAE = mae(actual_data, calculated_data)

print("The Mean Absolute Error between predicted and real consumptions on the training set is: {} W/m²".format(MAE))