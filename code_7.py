#If any of this libraries is missing from your computer. Please install them using pip.
#ARR_DELAY is the column name that should be used as dependent variable (Y).
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

filename = 'Flight_Delays_2018.csv'

#CODE
df = pd.read_csv(filename)


#DESCRIPTIVE STATISTICS
#creating a heatmap to find which variables have the highest correlation with delays initially
corr = df.corr(numeric_only = True).round(2)
sns.heatmap(corr, annot= True, vmax = 1, vmin = -0.5, cmap = 'icefire')
plt.show()


#Getting the descriptive statistics for the arrival delays 
#to compare with the statistics of arrival delays by airline and by airport to reduce the scope of data to analyze
print()
print("---Descriptive statistics for arrival delays---")
print(df['ARR_DELAY'].describe().round(2))


print()
print("---Descriptive statistics for arrival delays based on airline---")
print(df.groupby('OP_CARRIER_NAME')['ARR_DELAY'].agg(['mean','min','max']).sort_values(by ='mean', ascending=False).round(2))


print()
print("---Descriptive statistics for arrival delays based on airports---")
print(df.groupby('DEST')['ARR_DELAY'].agg(['mean','min','max']).sort_values(by='mean', ascending=False).head(10).round(2))


#Based on these results, I will reduce my scope by only including the 3 airports and 3 airlines with the highest delays
reduced = df.query("DEST=='ECP' or DEST =='SUN' or DEST == 'SHD'")
smol_df = df.query("OP_CARRIER_NAME=='Frontier Airlines Inc.' or OP_CARRIER_NAME=='JetBlue Airways' or OP_CARRIER_NAME=='Spirit Air Lines'")

#Creating an update correlation heatmap to see highest correlations with arrival delays after the reduction of the data
corr2 = reduced.corr(numeric_only = True).round(2)
sns.heatmap(corr2, annot= True, vmax = 1, vmin = -0.5, cmap = 'icefire')
plt.show()

corr3 = smol_df.corr(numeric_only = True).round(2)
sns.heatmap(corr3, annot= True, vmax = 1, vmin = -0.5, cmap = 'icefire')
plt.show()


#PREDICTIVE ANALYTICS, linear regression model and OLS

#Based on the results above, the predictor variables I selected are DEP_DELAY, CARRIER_DELAY & LATE_AIRCRAFT_DELAY
#None of the variables tested could provide a good model for predicting flight delays

x = df['DEP_DELAY']
y = df['ARR_DELAY']
x = sm.add_constant(x)
regression_model = sm.OLS(y,x).fit()
print(regression_model.summary())

a,b = np.polyfit(df['DEP_DELAY'],df['ARR_DELAY'],1)
plt.scatter(df['DEP_DELAY'],df['ARR_DELAY'], color='black')
plt.plot(df['DEP_DELAY'], a*df['ARR_DELAY']+b)
plt.text(250,1750, 'y = ' + '{:.3f}'.format(b) + ' + {:.3f}'.format(a) + 'x', size=12)
plt.xlabel('Departure Delay')
plt.ylabel('Arrival Delay')
plt.xlim(0,2000)
plt.ylim(0,2000)
plt.show()

# Sources
# OpenAI. (2025), Prompt: “How can I sort values from greatest to least when printing something from a df”. ChatGPT.

# OpenAI. (2025), Prompt: “How can I edit the x and y range in a scatter plot in python”. ChatGPT.