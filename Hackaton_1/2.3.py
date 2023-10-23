import pandas as pd
import scipy.stats as sc
from scipy import stats

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




# Calculer la production d'énergie pour chaque ville
daily_avg_caen_energy = daily_avg_caen.apply(lambda x: x*24*0.18*0.75).tolist()
daily_avg_tours_energy = daily_avg_tours.apply(lambda x: x*24*0.18*0.75).tolist()

print(daily_avg_caen_energy)
print(daily_avg_tours_energy)

variance_stat, p_value = stats.levene(daily_avg_caen_energy, daily_avg_tours_energy)

alpha = 0.05 # We can change this value depending on what we want 
if p_value > alpha:
    print(f"H_0 non rejettée. La p_valeur {p_value} est plus grande que notre alpha ({alpha}).")
else:
    print(f"H_0 rejettée. La p_valeur {p_value} est plus petite que notre alpha ({alpha}). Il y a donc une trop grosse différence entre les deux villes")