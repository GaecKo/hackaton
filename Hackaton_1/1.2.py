import pandas as pd
import math
from tabulate import tabulate

# ===== Load CSV & convert dates ===== #

df = pd.read_csv('Radiation.csv')

dates = pd.to_datetime(df['DATE'], format='%Y%m%d')

# ===== Caen data ===== #
caen_data = df['Caen']


 
# ===== Tours data ===== #
tours_data = df['Tours']



###########################################################################################################################
##       The following code calculates the number of days where the elecricity production has been encoded per month     ##
###########################################################################################################################

l=[]
l.append([dates[0].year, dates[0].month, 1])

months = []
months.append([dates[0].year,dates[0].month])

for i in range(1, len(dates)):
    
    if l[-1][0] == dates[i].year and l[-1][1] == dates[i].month:
        l[-1][2] += 1
        
    else:
        l.append([dates[i].year, dates[i].month, 1])
        months.append([dates[i].year,dates[i].month])

    
######################################################################
##       the average daily production of electricity in Wh/m2       ##
######################################################################
        
def average(data, l):
    """
        pre:
            > data: is a list containing the data for which we would like to calculate the average
            > l: is a list where each element is [the year, the month, the number of days where data was collected]
            
        post:
            > a list containing the average daily production of electricity in Wh/m2 for each month
        
    """
    count = 0
    means = []
    for j in l:
        somme = 0
        for k in range(count, count+j[2]):
            somme += data[k]
        means.append(somme/j[2])
        count += j[2]
    return means

#####################
##       end       ##
#####################


#####################################################################
##       the median daily production of electricity in Wh/m2       ##
#####################################################################

def median(data, l):
    """
        pre:
            > data: is a list containing the data for which we would like to calculate the average
            > l: is a list where each element is [the year, the month, the number of days where data was collected]
            
        post:
            > a list containing the median daily production of electricity in Wh/m2 for each month
        
    """
    count = 0
    medians = []
    for j in l :
        temp_list = []
        for k in range(count, count+j[2]):
            temp_list.append(data[k])
        temp_list.sort()
        medians.append(temp_list[j[2] // 2])
        count += j[2]
    return medians


#####################
##       end       ##
#####################





#################################################################################
##       the standard deviation daily production of electricity in Wh/m2       ##
#################################################################################

def standard_deviation(data, l, data_mean):
    """
        pre:
            > data: is a list containing the data for which we would like to calculate the average
            > l: is a list where each element is [the year, the month, the number of days where data was collected]
            > data_mean: the list returned by median(data, l)
            
        post:
            > a list containing the standard deviation daily production of electricity in Wh/m2 for each month
        
    """
    count = 0
    deviation = []
    for j in range(len(l)):
        somme = 0
        for k in  range(count, count+l[j][2]):
            somme += (data[k] - data_mean[j])**2
        deviation.append(math.sqrt(somme/l[j][2]))
        count += l[j][2]
    return deviation



#####################
##       end       ##
#####################

#######################################################################################
##       the 5% and 95% percentile of daily production of electricity in Wh/m2       ##
#######################################################################################

def percentile(percent, l, data):
    """
        pre:
            > data: is a list containing the data for which we would like to calculate the average.
            > l: is a list where each element is [the year, the month, the number of days where data was collected].
            > percent: the percentage for which we wish to calculate the percentile.
            
        post:
            > a list containing the percent's percentile of daily production of electricity in Wh/m2 for each month.
        
    """
    count = 0
    percentiles = []
    for j in range(len(l)):
        data_per_month = []
        for k in range(count, count+l[j][2]):
            data_per_month.append(data[k])
        d = pd.Series(data_per_month)
        percentile = d.quantile(percent/100)
        percentiles.append(percentile)
    return percentiles


caen_median = median(caen_data, l)

tours_median = median(tours_data, l)

caen_data_mean = average(caen_data, l)

tours_data_mean = average(tours_data, l)

caen_standard_deviation = standard_deviation(caen_data, l, caen_data_mean)

tours_standard_deviation = standard_deviation(tours_data, l, tours_data_mean)

caen_5_percentile = percentile(5, l, caen_data)

caen_95_percentile = percentile(95, l, caen_data)

tours_5_percentile = percentile(5, l, tours_data)

tours_95_percentile = percentile(95, l, tours_data)


data = {
    "Year": months,
    "Caen's median": caen_median,
    "Tours's median": tours_median,
    "Caen's mean": caen_data_mean,
    "Tours's mean": tours_data_mean,
    "Caen's standard deviation": caen_standard_deviation,
    "Tours's standard deviation": tours_standard_deviation,
    "Caen's 5% percentile": caen_5_percentile,
    "Caen's 95% percentile": caen_95_percentile,
    "Tours's 5% percentile": tours_5_percentile,
    "Tours's 95% percentile": tours_95_percentile
}

         
table = pd.DataFrame(data)

print(table)

