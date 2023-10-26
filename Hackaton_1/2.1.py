import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from scipy.stats import gamma, norm
import numpy as np


# ===== Load CSV & convert dates =====
df = pd.read_csv('Radiation.csv')

data = pd.read_csv('Radiation.csv')

data['DATE'] = pd.to_datetime(data['DATE'], format="%Y%m%d")

# Apply formula: C = E_Sol x 24 x P_cr x f_perf for both cities data
data['Caen'] = data['Caen']*24*0.18*0.75
data['Tours'] = data['Tours']*24*0.18*0.75


# parsing data to keep 1 line out of 100 (better visibility, less precision)
clean_data = data.iloc[range(0, len(data), 100), :]  

# C

# parsing data to keep 1 line out of 100 (better visibility, less precision)
clean_data = data.iloc[range(0, len(data), 100), :]  
# Donnée abérante de Caean
DataC = data.sort_values(by="Caen", axis=0)
Q1C = data["Caen"].quantile(0.25)
Q3C = data["Caen"].quantile(0.75)
MaxC = Q3C + 1.5 * (Q3C - Q1C)
MinC = Q1C - 1.5 * (Q3C - Q1C)

# Donnée abérante de Tours
DataT = data.sort_values(by="Tours", axis=0)
Q1T = data["Tours"].quantile(0.25)
Q3T = data["Tours"].quantile(0.75)
MaxT = Q3T + 1.5 * (Q3T - Q1T)
MinT = Q1T - 1.5 * (Q3T - Q1T)

# Variable avec données filtrées
DataParse = data[(DataC["Caen"] >= MinC) & (DataC["Caen"] <= MaxC) & (DataT["Tours"] >= MinT) & (DataT["Tours"] <= MaxT)]
#print(type(DataParse.values[0][1]))
#print(DataParse['DATE'])





dates = pd.to_datetime(DataParse['DATE'], format='%Y%m%d')
dates_avril = dates[dates.dt.month == 4]
#print(dates_avril)

# ===== Caen/Tours data ===== #
caen_data = df['Caen']
tours_data = df['Tours']

# ===== April data ===== #
april_caen = []
april_tours = []

for index in dates_avril.index:
    april_caen.append(caen_data[index]* 24 * 0.18 * 0.75)
    april_tours.append(tours_data[index]* 24 * 0.18 * 0.75)

#print(april_tours)

## 1) Fit Gamma and normal distributions by log-likelihood maximization to daily production of electricity during April (Caen & Tours)
print("===== 1 =====")  

# ===== Caen ===== #
# Fit Gamma distribution
gamma_params_caen = gamma.fit(april_caen, floc= -0.001)
print(gamma_params_caen)

# Fit Normal distribution
normal_params_caen = norm.fit(april_caen)
print(normal_params_caen)

# ===== Tours ===== #
# Fit Gamma distribution
gamma_params_tours = gamma.fit(april_tours, floc= -0.001)
print(gamma_params_tours)

# Fit Normal distribution
normal_params_tours = norm.fit(april_tours)
print(normal_params_tours)

## 2) Compute the 4 log-likelihoods and select the best model for each location (justify your answer).
print("===== 2 =====")

# ===== Caen ===== #
# Gamma
log_likelihood_gamma_caen = gamma.logpdf(april_caen, *gamma_params_caen).sum()
print(log_likelihood_gamma_caen)

# Normal
log_likelihood_normal_caen = norm.logpdf(april_caen, *normal_params_caen).sum()
print(log_likelihood_normal_caen)

# ===== Tours ===== #
# Gamma
log_likelihood_gamma_tours = gamma.logpdf(april_tours, *gamma_params_tours).sum()
print(log_likelihood_gamma_tours)

# Normal
log_likelihood_normal_tours = norm.logpdf(april_tours, *normal_params_tours).sum()
print(log_likelihood_normal_tours)

# Compare and select the best model
best_model_caen = "Gamma" if log_likelihood_gamma_caen > log_likelihood_normal_caen else "Normal"
best_model_tours = "Gamma" if log_likelihood_gamma_tours > log_likelihood_normal_tours else "Normal"

# Print and justify the best models
print(f"Best model for Caen: {best_model_caen} (Log-Likelihood: {log_likelihood_gamma_caen if best_model_caen == 'Gamma' else log_likelihood_normal_caen})")
print(f"Best model for Tours: {best_model_tours} (Log-Likelihood: {log_likelihood_gamma_tours if best_model_tours == 'Gamma' else log_likelihood_normal_tours})")


## 3) Compare on the same plot the empirical, the gamma and normal pdf (the empirical pdf is an histogram of frequencies)

# ===== Caen ===== #
# Gamma
gamma_x_caen = np.linspace(0, 1000, 1000)
gamma_y_caen = gamma.pdf(gamma_x_caen, *gamma_params_caen)

# Normal
normal_x_caen = np.linspace(0, 1000, 1000)
normal_y_caen = norm.pdf(normal_x_caen, *normal_params_caen)

# ===== Tours ===== #
# Gamma
gamma_x_tours = np.linspace(0, 1000, 1000)
gamma_y_tours = gamma.pdf(gamma_x_tours, *gamma_params_tours)

# Normal
normal_x_tours = np.linspace(0, 1000, 1000)
normal_y_tours = norm.pdf(normal_x_tours, *normal_params_tours)

# ===== Plot ===== #

# Caen
plt.figure()
plt.hist(april_caen, bins=100, density=True, label='Empirical')
plt.plot(gamma_x_caen, gamma_y_caen, label='Gamma')
plt.plot(normal_x_caen, normal_y_caen, label='Normal')
plt.legend()
plt.title('Caen')
plt.show()

# Tours
plt.figure()
plt.hist(april_tours, bins=100, density=True, label='Empirical')
plt.plot(gamma_x_tours, gamma_y_tours, label='Gamma')
plt.plot(normal_x_tours, normal_y_tours, label='Normal')
plt.legend()
plt.title('Tours')
plt.show()
