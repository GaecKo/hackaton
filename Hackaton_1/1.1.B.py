import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# récup les données
data = pd.read_csv('Radiation.csv')
data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m%d')
data = data.iloc[range(0, len(data), 100), :]  # parsing de data pour garde 1 ligne sur 100. (+ lisible, - précis)
# 0 10 25 50 100


x = [data['Caen']*24*0.18*0.75,
     data['Tours']*24*0.18*0.75]
df = pd.DataFrame(x, index=['Caen', 'Tour'])

df.T.boxplot(vert=False)
plt.subplots_adjust(left=0.25)
plt.show()