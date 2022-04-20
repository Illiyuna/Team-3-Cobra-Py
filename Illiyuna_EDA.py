# Team 3 -- Cobra Py
# Illiyuna Code 

#%%
import csv
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# %%

#from pathlib import Path  
#filepath = Path('/Users/illiyunaislam/OneDrive/Grad School/Courses/Data Mining/Team-3-Cobra-Py/')  
# filepath.parent.mkdir(parents=True, exist_ok=True)


# %%
def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

# %%
#Subsetting loans for only non-commercial business homes
#This variable can be dropped
hmda=filter_rows_by_values(hmda, "business_or_commercial_purpose", [1,1111])
bs=hmda[["business_or_commercial_purpose"]]
bs.business_or_commercial_purpose.value_counts()


# %%
#subsetting loans for loans that are under 950k to capture single family homes

la=hmda[['loan_amount']]
la.loan_amount.value_counts()
hmda1=hmda[hmda['loan_amount']<950000]
la=hmda1[['loan_amount']]
la.loan_amount.value_counts()

# %%
#Subsetting for, getting rid of multifamily properties
#This variable can be dropped
mf =hmda1[['multifamily_affordable_units']]
mf.multifamily_affordable_units.value_counts()
hmda1=filter_rows_by_values(hmda1, 'multifamily_affordable_units', ['0', '100', 0.0, 100.0,])
mf =hmda1[['multifamily_affordable_units']]
mf.multifamily_affordable_units.value_counts()

# %%
#Subsetting for Primary resident properties/ getting rid of non-primary residence
#This variable can be dropped
ot=hmda1[['occupancy_type']]
ot.occupancy_type.value_counts()

hmda1=filter_rows_by_values(hmda1, 'occupancy_type', [2,3])
ot=hmda1[['occupancy_type']]
ot.occupancy_type.value_counts()
# %%
#Subsetting for total units equal to or less than 4 / getting rid of total units greater than or equal to 5
tu=hmda1[['total_units']]
tu.total_units.value_counts()

hmda1=filter_rows_by_values(hmda1, 'total_units', ['5-24','>149','25-49','100-14','50-99'])
tu=hmda1[['total_units']]
tu.total_units.value_counts()



# %%
#Subsetting for 30 year loans if we want to capture 15 year loans get rid of 180 below
#This variable can be dropped
lt=hmda1[['loan_term']]
lt.loan_term.value_counts()

# %%
#This variable can be dropped
hmda1=filter_rows_by_values(hmda1, 'loan_term', ['Exempt'])
hmda1["loan_term"] = pd.to_numeric(hmda1["loan_term"])

hmda1=hmda1[hmda1['loan_term']==360]
lt=hmda1[['loan_term']]
lt.loan_term.value_counts()

# %%
#Subsetting for purchase loans
#This variable can be dropped
ltype=hmda1[['loan_purpose']]
ltype.loan_purpose.value_counts()
hmda1=hmda1[hmda1['loan_purpose']==1]
ltype=hmda1[['loan_purpose']]
ltype.loan_purpose.value_counts()

############################

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


# Provide a two-way table of the action taken on the loan by preapproval.


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