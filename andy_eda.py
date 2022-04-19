#%%

#############
## Imports ##
#############

from cgi import test
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
    'derived_dwelling_category', #
    'derived_ethnicity',
    'derived_race',
    'derived_sex',
    'action_taken',
    'loan_purpose',
    'business_or_commercial_purpose',
    'occupancy_type', # construction_method
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
        #new_val=float(int(new_val))
        new_val=int(new_val)
        #print(f'{type(new_val)} {new_val}')
        return (new_val if new_val>=0 else np.nan) 
    except:
        #Catch all for situations not covered
        return np.nan

##################################################

def clean_int_rate(val):
    
    #remove any whitespace
    new_val=val.strip()
    
    #Attempt to convert directly to float, returning NaN if negative
    try:
        new_val=float(val)
        return (new_val if new_val>=0 else np.nan)
    except:
        pass
    
    #Convert interest rate to 0, if exempt
    if val=='Exempt':
        return 0
    
    #Catch all for situations not covered
    return np.nan

##################################################

#%%

#pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
#Subset for desired test data
col_names1=["property_value", "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
col_names2=["property_value", "derived_dwelling_category", "occupancy_type", "construction_method", "total_units", "interest_rate", "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
test_df=dc_df[col_names2].copy()

#Change property_value column to numerical
test_df.iloc[:,0:1]=test_df.loc[:, 'property_value'].apply(clean_propval)#.astype('Int64')
#test_df.iloc[:,0:1]=test_df.iloc[:,0:1].astype('Int64')
test_df.head()

#%%
#graphing
#Supressing Log tick label
#https://stackoverflow.com/questions/68468307/how-do-i-change-le6-to-1000000-in-matplotlib

def make_scatter(data, x, y):
    sns.regplot(data=data, x=x, y=y, scatter_kws={'alpha':0.03})
    plt.ticklabel_format(style='plain')
    plt.xticks(rotation=45)
    plt.show()

#Filter for extreme prop val
test_df=test_df.dropna()
cond=test_df['property_value']<=2000000
filtered_popval=test_df[cond]

#hist of prop val
#g=sns.histplot(binrange=(0,2000000), binwidth=10000, data=test_df, x='property_value')
#g.set_xticklabels(['', '0', '200,000', '400,000', '600,000', '800,000', '1,000,000'])
g=sns.histplot(data=filtered_popval, binwidth=10000, x='property_value')
g.ticklabel_format(style='plain')
plt.xticks(rotation=45)
plt.show()


#%%
#scatterplot: val against pop
# g=sns.scatterplot(data=filtered_popval, x='tract_population', y='property_value')
# g.ticklabel_format(style='plain')
# plt.xticks(rotation=45)
# plt.show()
make_scatter(filtered_popval.iloc[:, np.r_[0, 6]], x='tract_population', y='property_value')
make_scatter(filtered_popval, x='tract_minority_population_percent', y='property_value')
make_scatter(filtered_popval, x='ffiec_msa_md_median_family_income', y='property_value')
make_scatter(filtered_popval, x='tract_to_msa_income_percentage', y='property_value')
make_scatter(filtered_popval, x='tract_owner_occupied_units', y='property_value')
make_scatter(filtered_popval, x='tract_one_to_four_family_homes', y='property_value')
make_scatter(filtered_popval, x='tract_median_age_of_housing_units', y='property_value')

#%%
#GRAPH TESTING FOR PROP VAL -- isna ERRORS -- was due to making int instead of float?

#Get slice
# col_names1=["property_value", "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
# test_slice=dc_df.loc[0:1000, col_names1].copy()
# #Clean slice
# test_slice['property_value']=test_slice['property_value'].apply(clean_propval)
# #Regplot slice
# sns.regplot(data=test_slice, x='tract_population', y='property_value')
# plt.show()

#%%

#derived_dwelling_category + occupancy_type + construction_method + total_units + interest_rate


#Build test linear model
model_test=ols(formula='property_value ~ C(derived_dwelling_category) + C(occupancy_type) + C(construction_method) + C(total_units) + interest_rate + tract_population + tract_minority_population_percent + ffiec_msa_md_median_family_income + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units', data=test_df)
model_test_fit=model_test.fit()
print(model_test_fit.summary())
# %%
