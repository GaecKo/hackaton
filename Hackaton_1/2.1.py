import pandas as pd

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

print(april_tours)
