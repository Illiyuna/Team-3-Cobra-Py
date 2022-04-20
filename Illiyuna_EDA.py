# Team 3 -- Cobra Py
# Illiyuna Code 

#%%
import csv
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# data dictionary link below:
#https://ffiec.cfpb.gov/documentation/2018/lar-data-fields/

#%% Used to unzip file

# from zipfile import ZipFile
# with ZipFile('2018_public_lar_csv.zip', 'r') as zipObject:
#    listOfFileNames = zipObject.namelist()
#    for fileName in listOfFileNames:
#        if fileName.endswith('.csv'):
#            # Extract a single file from zip
#            zipObject.extract(fileName, 'temp_py')
#            print('All the python files are extracted')

os.getcwd()
os.chdir('/Users/illiyunaislam/OneDrive/Grad School/Courses/Data Mining/Team-3-Cobra-Py')
os.getcwd()

#%%
# Import Data 
df = pd.read_csv('dc_df.csv')


# %%
hmda=df[df.columns[df.columns.isin(['activity_year', 
                                    'lei', 
                                    'state_code',
                                    'census_tract', 
                                    'derived_loan_product_type',
                                    'derived_ethnicity', 
                                    'derived_race', 
                                    'derived_sex', 
                                    'action_taken',
                                    'preapproval', 
                                    'loan_type', 
                                    'loan_purpose',
                                    'business_or_commercial_purpose', 
                                    'loan_amount',
                                    'combined_loan_to_value_ratio', 
                                    'interest_rate',
                                    'loan_term',
                                    'property_value', 
                                    'occupancy_type',
                                    'total_units', 
                                    'multifamily_affordable_units', 
                                    'income', 
                                    'debt_to_income_ratio',
                                    'applicant_age', 
                                    'co_applicant_age', 
                                    'tract_population', 
                                    'tract_minority_population_percent'])]]


#%%
df.head()
# %%
# Summary of Loan Amount By Loan Type

result = df.groupby('loan_type').agg({'loan_amount': ['mean', 'min', 'max']})
  
print("Mean, Min, and max values of Loan Amounts By Type")
print(result)

# chnage numbers to actual loan names 

#%%
#A grouped bar chart with the average loan value by loan 
# type. Label the averages for each bar on the chart.

# Used construction method as a proxy from property type 
# Loan Type
# 1 -- Conventional (not insured or guaranteed by FHA VA RHS or FSA)
# 2 -- Federal Housing Administration insured (FHA)
# 3 -- Veterans Affairs guranteed (VA)
# 4 -- USDA Rural Housing Service or Farm Service Agency guaranteed (RHS or FSA)

sns.barplot(data=df, 
            x='loan_type', 
            y='loan_amount', 
            hue='construction_method', ci=None)
plt.xlabel('Type of Loan',size=14)
plt.ylabel('Average Loan Amount',size=14)
plt.title('Average Loan Amount & Type by Property Type') 
x_var= ['Conventional', 
        'FHA Insured', 
        'VA Insured',
        'RHS or FSA Insured']
plt.xticks([0,1,2,3], x_var, rotation=20)
labels= ['Site Built', 'Manufactured Home']         
plt.legend(labels,loc='upper left')
plt.show()

#%%
# Average values of loans for 2020 by Loan Type.

df.groupby(['loan_type'], as_index=False).mean().groupby('loan_type')['loan_amount'].mean()

#%%
# Average values of loans by construction method.
df.groupby(['construction_method'], as_index=False).mean().groupby('construction_method')['loan_amount'].mean()

#%%
# Average values of loans  by occupancy type.
df.groupby(['occupancy_type'], as_index=False).mean().groupby('occupancy_type')['loan_amount'].mean()


#%%

# Lets look at action taken 

ncount = len(df)
plt.figure(figsize=(12,8))
ax= sns.countplot(x="action_taken", data=df)
plt.title('Distribution of Truck Configurations')
plt.xlabel('Number of Axles')
# Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Frequency [%]')

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text

# Use a LinearLocator to ensure the correct number of ticks
ax.yaxis.set_major_locator(ticker.LinearLocator(11))

# Fix the frequency range to 0-100
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
ax2.grid(None)

plt.show


# okay so problem we have a houston,

# no counts for 7 or 8 so Q2 is out the window. Sample size too small. 

#%%
# Provide a two-way table of the action taken on the loan by preapproval.

grouped = df.groupby(['action_taken', 'preapproval'])
grouped.size()
# ii. Run a Chi-Squared test on the frequency table. Interpret the result.
# iii. Examine the “Not Applicable” category. What percentage of the data falls into
# this category?
# iv. Set “Not Applicable” to be missing and rerun the Chi-Squared test. Does the
# result and interpretation change? Is this test meaningful?

#%%
# Examine the variable “Action Taken” by the following variables. For each of the variables,
# examine the crosstabulations and provide a meaningful graph that shows the differences. For
# each entry, do you see any differences for the groups that are either “Application Denied” or
# “Preapproval request denied”?
# a. Income
# b. Race of Applicant (1)
# c. Sex
# d. Lien Status

#Is there a statistically significant difference between the “pre-approval denied”
# and “application denied” groups for Income, Race, Sex, and Lien Status?