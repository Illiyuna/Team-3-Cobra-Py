#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


#%%
import csv
#data= pd.read_csv('2018_public.csv')

# %%
data= pd.read_csv('out.csv')
df=data
df.head()
df.columns.values


# %%
hmda=df[df.columns[df.columns.isin(['activity_year', 'lei', 'state_code','census_tract', 'derived_loan_product_type','derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken','preapproval', 'loan_type', 'loan_purpose','business_or_commercial_purpose', 'loan_amount','combined_loan_to_value_ratio', 'interest_rate','loan_term','property_value', 'occupancy_type','total_units', 'multifamily_affordable_units', 'income', 'debt_to_income_ratio','applicant_age', 'co_applicant_age', 'tract_population', 'tract_minority_population_percent'])]]


# %%

from pathlib import Path  
filepath = Path('C:/Users/brand/OneDrive/Desktop/DM Final Project/out.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)




# %%
hmda.to_csv(filepath)
# %%
hmda.head()
hmda.columns.values
# %%
#Checking Year is 2018
#This variable can be dropped
act_year=hmda[["activity_year"]].apply(pd.to_numeric)
act_year.activity_year.value_counts()


# %% Filtering out State varibles that are not correct /do not align with a state
state_code=hmda[["state_code"]]
state_code.state_code.value_counts()

# %%
def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]
# %%
hmda=filter_rows_by_values(hmda, "state_code", ["AS","MH"])
state_code=hmda[["state_code"]]
state_code.state_code.value_counts()


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
mf=hmda1[['multifamily_affordable_units']]
mf.multifamily_affordable_units.value_counts()
hmda1=filter_rows_by_values(hmda1, 'multifamily_affordable_units', ['0', '100', 0.0, 100.0,])
mf=hmda1[['multifamily_affordable_units']]
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
# %%
