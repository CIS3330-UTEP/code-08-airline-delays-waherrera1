#If any of this libraries is missing from your computer. Please install them using pip.
#ARR_DELAY is the column name that should be used as dependent variable (Y).
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'Flight_Delays_2018.csv'

#CODE

df = pd.read_csv(filename)
df.describe()

#creating a heatmap to find which numerical variables are correlated the most with the flight delays variable
#which reveals the variables with the highest correlation to delays are departure delay and carrier delay
corr = df.corr(numeric_only = True).round(2)
sns.heatmap(corr, annot= True, vmax = 1, vmin = -0.5, cmap = 'icefire')

#Getting the descriptive statistics and creating a scatter plot to compare airports based on flight delays
airlines = df.groupby('OP_CARRIER_NAME')['ARR_DELAY']
airlines.describe()

df.plot.scatter(x='OP_CARRIER_NAME', y='ARR_DELAY')
plt.show()

#Creating a scatter plot to compare airlines based on flight delays
airlines = df.groupby('ORIGIN')['ARR_DELAY']
airlines.describe()

df.plot.scatter(x='ORIGIN', y='ARR_DELAY')
plt.show()

#Based on these results, I will remove the airlines and airports that experience the least delays to focus on data that is more significant to delays


