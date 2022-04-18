#%%

#############
## Imports ##
#############

from operator import index
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import re

from statsmodels.formula.api import ols


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

##########################
## Set Display Settings ##
##########################

#DF Settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

#Graph Color Scheme
color_list=['BuPu', 'PRGn', 'Pastel1', 'Pastel2', 'Set2', 'binary', 'bone', 'bwr',
                 'bwr_r', 'coolwarm', 'cubehelix', 'gist_earth', 'gist_gray', 'icefire',]

# for color in color_list:
#     print(color)
#     g = sns.FacetGrid(df_all, col="world", hue='world', palette=color)
#     g.map_dataframe(sns.histplot, x="income00")
#     plt.show()

#Graphs
sns.set_palette('Set2')

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%
#Load in dc_df
dc_df=pd.read_csv("data/dc_df.csv")

#%%
# show number of obs and display df head
print(f'Dc observations: {len(dc_df)}')
display(dc_df.head().style.set_sticky(axis="index"))

#%%
#Get variables interested in

#Cat vars I am interested in
categorical_variabless_list=[
    'derived_loan_product_type',
    'derived_dwelling_category',
    'derived_ethnicity',
    'derived_race',
    'derived_sex',
    'action_taken',
    'loan_purpose',
    'business_or_commercial_purpose',
    'occupancy_type',
    'aus_1',
    'aus_2',
    'aus_3',
    'aus_4',
    'aus_5',
    'denial_reason_1',
    'denial_reason_2',
    'denial_reason_3',
    'denial_reason_4']

continuous_variabless_list = [
    'loan_amount',
    'interest_rate',
    'loan_term',
    'property_value',
    'income',
    'debt_to_income_ratio',
    'applicant_age',
    'co_applicant_age',
    'tract_population',
    'tract_minority_population_percent',
    'ffiec_msa_md_median_family_income',
    'tract_to_msa_income_percentage',
    'tract_owner_occupied_units',
    'tract_one_to_four_family_homes',
    'tract_median_age_of_housing_units']

#%%
#look at summary statistics


#%%

###############
## Functions ##
###############

def clean_propval_col(prop_val):

    #Get the propety value for the given row
    #prop_val = row['property_value']
    
    #Attempt to turn it into an integer
    try: 
        prop_val = int(prop_val)
    except: 
        pass

    #Attempt to turn it into a float
    try: 
        if not isinstance(prop_val,int): 
            prop_val = float(prop_val)
    except: 
        pass

    #If numeric and not a boolean, return the value -- return Nan if value is less than 0 or is a boolean
    if ( isinstance(prop_val,int) or isinstance(prop_val,float) ) and not isinstance(prop_val, bool):
        return ( prop_val if prop_val>=0 else np.nan )
    if isinstance(prop_val, bool):
        return np.nan

    #If dealing with a complex string, adjust for approriate numeric value

    # #Get rid of any whitespace
    # prop_val = prop_val.strip()
    # #Assign values depending on string content
    # if thechildren == "Dk na":
    #     return np.nan
    # if thechildren == "Eight or more": 
    #     thechildren = min(8 + np.random.chisquare(2) , 12)
    #     return thechildren # leave it as decimal
    
    #Catch All for not covered situations
    return np.nan

##################################################

def clean_propval(val):
    
    try:
        #Attempt to identify and extract the integer value
        regex=r'^[0-9]+'
        new_val=re.findall(regex, val)[0]
        #Convert to integer then float and return if positive, Nan if less than 0
        new_val=float(int(new_val))
        return (new_val if new_val>=0 else np.nan) 
    except:
        #Catch all for not situations not covered
        return np.nan

##################################################

#%%

pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
#Subset for desired test data
col_names=["property_value", "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
test_df=dc_df[col_names].copy()

#Change property_value column to numerical
test_df.iloc[:,0:1]=test_df.loc[:, 'property_value'].apply(clean_propval)
test_df.head()


#%%
#Build test linear model
model_test=ols(formula='property_value ~ tract_population + tract_minority_population_percent + ffiec_msa_md_median_family_income + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units', data=test_df)
model_test_fit=model_test.fit()
print(model_test_fit.summary())
# %%
