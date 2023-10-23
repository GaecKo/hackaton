import pandas as pd
import scipy.stats as sc
import numpy as np


# ===== Load CSV & convert dates ===== #

df = pd.read_csv('Radiation.csv')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')

# ===== Caen/Tours data ===== #
# Filtrer les données pour le mois d'avril
april_data = df[df['DATE'].dt.month == 4]

caen_data = april_data['Caen']
tours_data = april_data['Tours']

# Calculer les moyennes quotidiennes pour chaque ville
daily_avg_caen = caen_data.groupby(april_data['DATE'].dt.day).mean()
daily_avg_tours = tours_data.groupby(april_data['DATE'].dt.day).mean()


# Convertir les résultats en listes
daily_avg_caen_list = daily_avg_caen.tolist()
daily_avg_tours_list = daily_avg_tours.tolist()

# Calculer la production d'énergie pour chaque ville
daily_avg_caen_energy = daily_avg_caen.apply(lambda x: x*24*0.18*0.75)
daily_avg_tours_energy = daily_avg_tours.apply(lambda x: x*24*0.18*0.75)

# Calculer la moyenne
caen_april_energy_mean = daily_avg_caen_energy.mean()

tours_april_energy_mean = daily_avg_tours_energy.mean()

n = 30

# Calculer S_caen et S_tours

S_caen = np.std(daily_avg_caen_energy.tolist()) 
S_tours = np.std(daily_avg_tours_energy.tolist()) 

print(S_caen)

S_pool = np.sqrt((S_caen*(n-1) + S_tours*(n-1)) / (n+n-2))



# Calculer S_pool 

# Calcul de t_obs ~ t_(n1+n2-2)
t_obs = (caen_april_energy_mean - tours_april_energy_mean)/( (S_pool) * (((1/n) + 1/n)**(1/2)))

print("T_obs = ", t_obs)

alpha = 0.05

t_l = sc.t.ppf(q= (alpha / 2), df=n+n+2)
t_u = sc.t.ppf(q= (1-(alpha/2)), df=n+n+2)

print(t_l, " < ", t_obs, " < ", t_u)