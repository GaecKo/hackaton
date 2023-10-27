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

#print(result.summary())

#3. judge quality

actual_data = training_set['Caen']
calculated_data = result.predict()

# ===== Check for NaN values ===== #

actual_data_nan = actual_data.isnull().any()
calc_data_nan = calculated_data.isnull().any()

# ===== Remove the missing values and make the sets the same size ===== #

missing_rows = actual_data.isnull() | calculated_data.isnull()
actual_data = actual_data[~missing_rows]
calculated_data = calculated_data[~missing_rows]

plt.plot(training_set['YEAR'], actual_data/10, label="Measured Temp.", linestyle="-", color="blue")
plt.plot(training_set['YEAR'], calculated_data.values/10, label="Predicted Temp.", linestyle="--", color="red")

plt.title("Measured temperature and predicted temperature by AutoReg model")
plt.xlabel("Date")
plt.ylabel("Temperature (celsius)")
plt.legend()
#plt.show()

#4. compute the Mean Absolute error (MAE) on the training set

# ===== Calculate the MAE ===== #

MAE = mae(actual_data, calculated_data)

print("The Mean Absolute Error between predicted and real consumptions on the training set is: {} W/m²".format(MAE))

#3.3

from sklearn.metrics import r2_score

#Compare on a graph, the forecast to  real consumptions on the given period
model2 = AutoReg(validation_set['Caen'], lags=10)
result2 = model.fit()

actual_data1 = validation_set['Caen']
calculated_data1 = result2.predict()

# ===== Check for NaN values ===== #

actual_data_1nan = actual_data1.isnull().any()
calc_data_1nan = calculated_data1.isnull().any()

# ===== Remove the missing values and make the sets the same size ===== #

if actual_data_1nan | calc_data_1nan:
    missing_rows = actual_data1.isnull() | calculated_data1.isnull()
    actual_data1 = actual_data1[~missing_rows]
    calculated_data1 = calculated_data1[~missing_rows]

plt.plot(validation_set['YEAR'], actual_data1/10, label="Measured Temp.", linestyle="-", color="blue")
plt.plot(validation_set['YEAR'], calculated_data1.values/10, label="Predicted Temp.", linestyle="--", color="red")

plt.title("Measured temperature and predicted temperature by AutoReg model")
plt.xlabel("Date")
plt.ylabel("Temperature (celsius)")
plt.legend()
plt.show()

#Plot the errors of prediction. Are they acceptable?

prediction_error = actual_data1 - calculated_data1
plt.figure(figsize=(12,12))
plt.plot(validation_set['YEAR'], prediction_error, label="Prediction errors", color="green")
plt.title("Prediction Errors on the validation set")
plt.xlabel("Date")
plt.ylabel("Error")
plt.legend()
plt.show()

#Compute the MAE on the test set and the R². Is the forecast reliable?

mae2 = mae(actual_data1, calculated_data1)
r_squared_val = r2_score(actual_data1, calculated_data1)

print("The MAE and the R² values are respectively: {} and {}".format(mae2, r_squared_val))
