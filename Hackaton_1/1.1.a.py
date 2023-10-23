import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===== Load CSV & convert dates =====

data = pd.read_csv('Radiation.csv')

data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m%d')

# parsing data to keep 1 line out of 100 (better visibility, less precision)
data = data.iloc[range(0, len(data), 100), :]  

# ===== Plot settings =====

# taille
plt.figure(figsize=(36, 12))
plt.xlim(pd.Timestamp('1974-01-01'), pd.Timestamp('2023-12-31'))

# étiquette arrière plan
plt.grid(True)
years = mdates.YearLocator()
year_fmt = mdates.DateFormatter('%Y')
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(year_fmt)
plt.xticks(rotation=45)

# légende
plt.xlabel('Date')
plt.ylabel('solar electric production (WH/m²)')
plt.title('solar electric production in Caen and Tours from 1974 to 2023')
plt.legend()

# ===== Plot =====

# Apply formula: C = E_Sol x 24 x P_cr x f_perf for both cities data
plt.plot(data['DATE'], data['Caen']*24*0.18*0.75, label='Caen', color='darkgoldenrod')

plt.fill_between(data['DATE'], data['Caen']*24*0.18*0.75, color='darkgoldenrod', alpha=0.3)

plt.plot(data['DATE'], data['Tours'], label='Tours', color='darkblue')


# ===== Show Plot =====
plt.show()
