import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# récup les données
data = pd.read_csv('Radiation.csv')
data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m%d')
data = data.iloc[range(0, len(data), 100), :]  # parsing de data pour garde 1 ligne sur 100. (+ lisible, - précis)
# 0 10 25 50 100

# taille
plt.figure(figsize=(36, 12))
plt.xlim(pd.Timestamp('1974-01-01'), pd.Timestamp('2023-12-31'))

# étiquette arrière plan
plt.grid(True)
years = mdates.YearLocator()
year_fmt = mdates.DateFormatter('%Y')
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(year_fmt)

# Trace les courbes
plt.plot(data['DATE'], data['Caen']*24*0.18*0.75, label='Caen', color='darkgoldenrod')
plt.fill_between(data['DATE'], data['Caen']*24*0.18*0.75, color='darkgoldenrod', alpha=0.3)
plt.plot(data['DATE'], data['Tours'], label='Tours', color='darkblue')

# légende
plt.xlabel('Date')
plt.ylabel('solar electric production')
plt.title('solar electric production in Caen and Tours from 1974 to 2023')
plt.legend()

plt.show()
