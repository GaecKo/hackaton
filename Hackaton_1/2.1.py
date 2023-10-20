import pandas as pd
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt

# ===== Load CSV & convert dates ===== #

df = pd.read_csv('Radiation.csv')

dates = pd.to_datetime(df['DATE'], format='%Y%m%d')
dates_avril = dates[dates.dt.month == 4]

# ===== Caen/Tours data ===== #
caen_data = df['Caen']
tours_data = df['Tours']

# ===== April data ===== #
april_caen = []
april_tours = []

for index in dates_avril.index:
    april_caen.append(caen_data[index])
    april_tours.append(tours_data[index])

#print(april_tours)

## 1) Fit Gamma and normal distributions by log-likelihood maximization to daily production of electricity during April (Caen & Tours)
print("===== 1 =====")  

# ===== Caen ===== #
# Fit Gamma distribution
gamma_params_caen = gamma.fit(april_caen, floc= -0.001)
print(gamma_params_caen)

# Fit Normal distribution
mean_caen = np.mean(april_caen)
std_caen = np.std(april_caen)
normal_params_caen = (mean_caen, std_caen)
print(normal_params_caen)

# ===== Tours ===== #
# Fit Gamma distribution
gamma_params_tours = gamma.fit(april_tours, floc= -0.001)
print(gamma_params_tours)

# Fit Normal distribution
mean_tours = np.mean(april_tours)
std_tours = np.std(april_tours)
normal_params_tours = (mean_tours, std_tours)
print(normal_params_tours)

## 2) Compute the 4 log-likelihoods and select the best model for each location (justify your answer).
print("===== 2 =====")

# ===== Caen ===== #
# Gamma
log_likelihood_gamma_caen = gamma.logpdf(april_caen, *gamma_params_caen).sum()
print(log_likelihood_gamma_caen)

# Normal
log_likelihood_normal_caen = gamma.logpdf(april_caen, *normal_params_caen).sum()
print(log_likelihood_normal_caen)

# ===== Tours ===== #
# Gamma
log_likelihood_gamma_tours = gamma.logpdf(april_tours, *gamma_params_tours).sum()
print(log_likelihood_gamma_tours)

# Normal
log_likelihood_normal_tours = gamma.logpdf(april_tours, *normal_params_tours).sum()
print(log_likelihood_normal_tours)


## 3) Compare on the same plot the empirical, the gamma and normal pdf (the empirical pdf is an histogram of frequencies)

# ===== Caen ===== #
# Gamma
gamma_x_caen = np.linspace(0, 1000, 1000)
gamma_y_caen = gamma.pdf(gamma_x_caen, *gamma_params_caen)

# Normal
normal_x_caen = np.linspace(0, 1000, 1000)
normal_y_caen = gamma.pdf(normal_x_caen, *normal_params_caen)

# ===== Tours ===== #
# Gamma
gamma_x_tours = np.linspace(0, 1000, 1000)
gamma_y_tours = gamma.pdf(gamma_x_tours, *gamma_params_tours)

# Normal
normal_x_tours = np.linspace(0, 1000, 1000)
normal_y_tours = gamma.pdf(normal_x_tours, *normal_params_tours)

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

