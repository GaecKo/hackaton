import pandas

Data = pandas.read_csv('Radiation.csv')
Data["Caen"] = Data["Caen"] * 24 * 0.18 * 0.75
Data["Tours"] = Data["Tours"] * 24 * 0.18 * 0.75

# Donnée abérante de Caean
DataC = Data.sort_values(by="Caen", axis=0)
Q1C = Data["Caen"].quantile(0.25)
Q3C = Data["Caen"].quantile(0.75)
MaxC = Q3C + 1.5 * (Q3C - Q1C)
MinC = Q1C - 1.5 * (Q3C - Q1C)

# Donnée abérante de Tours
DataT = Data.sort_values(by="Tours", axis=0)
Q1T = Data["Tours"].quantile(0.25)
Q3T = Data["Tours"].quantile(0.75)
MaxT = Q3T + 1.5 * (Q3T - Q1T)
MinT = Q1T - 1.5 * (Q3T - Q1T)

# Variable avec donnée filtrée
DataParse = Data[(DataC["Caen"] >= MinC) & (DataC["Caen"] <= MaxC) & (DataT["Tours"] >= MinT) & (DataT["Tours"] <= MaxT)]

