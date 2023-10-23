import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from tabulate import tabulate

# ===== Load CSV & convert dates =====

data = pd.read_csv('Radiation.csv')

data['DATE'] = pd.to_datetime(data['DATE'], format="%Y%m%d")

# Apply formula: C = E_Sol x 24 x P_cr x f_perf for both cities data
data['Caen'] = data['Caen']*24*0.18*0.75
data['Tours'] = data['Tours']*24*0.18*0.75

df = pd.read_csv('Radiation.csv')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')

# Filtrer les données pour le mois d'avril
april_data = data[data['DATE'].dt.month == 4 & (df["DATE"].dt.year >= 1977) & (data["DATE"].dt.year <= 2019)]

caen_data = april_data['Caen']
tours_data = april_data['Tours']

# Calculer les moyennes quotidiennes pour chaque ville
daily_avg_caen = caen_data.groupby(april_data['DATE'].dt.day).mean()
daily_avg_tours = tours_data.groupby(april_data['DATE'].dt.day).mean()

print(caen_data)
print(daily_avg_tours)