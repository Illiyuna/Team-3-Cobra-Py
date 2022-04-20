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
from statsmodels.stats.outliers_influence import variance_inflation_factor

#import os

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

#RGB values of pallette
#print(sns.color_palette('Set2').as_hex())
#col_pallette=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

#Graphs
sns.set_palette('Set2')

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

###############
## Functions ##
###############

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

def make_scatter(data, x, y):
    ax1=sns.regplot(data=data, x=x, y=y, color="#b3b3b3", scatter_kws={'alpha':0.03})
    ax1.lines[0].set_color("#e5c494")
    plt.ticklabel_format(style='plain')
    plt.xticks(rotation=45)
    plt.show()

##################################################

def make_hist_kde(data, x, bins):
    #change line color 
    #source: https://github.com/mwaskom/seaborn/issues/2344
    
    #Initialize graph
    ax1=sns.histplot(data=data, x=x, bins=bins, color="#8da0cb",
                     kde=True, line_kws=dict(linewidth=4))
    #Change color of KDE line
    ax1.lines[0].set_color("#e78ac3")
    #Format and display
    plt.ticklabel_format(style='plain')
    plt.xticks(rotation=45)
    plt.show()

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

#pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
#Subset for desired test data
col_names1=["property_value", "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
col_names2=["property_value", "derived_dwelling_category", "occupancy_type", "construction_method", "total_units", "interest_rate", "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
test_df=dc_df[col_names2].copy()

#Change property_value column to numerical
test_df.iloc[:,0:1]=test_df.loc[:, 'property_value'].apply(clean_propval)#.astype('Int64')

#Filter for houses below 1.25 mil
cut_off=2000000
cond=test_df['property_value']<=cut_off
filtered_popval=test_df[cond]

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#Summary Statistics
display(filtered_popval.head().style.set_sticky(axis="index"))
display(filtered_popval.describe().style.set_sticky(axis="index"))
print()
display(filtered_popval.derived_dwelling_category.value_counts())
print()
display(filtered_popval.occupancy_type.value_counts())
print()
display(filtered_popval.construction_method.value_counts())
print()
display(filtered_popval.total_units.value_counts())

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%
#Bar graphs

#%%
#Box plots

#%%
#hist plots of all variables examined in this question
#binw=None
strg=16
rice=36
bins=strg
make_hist_kde(filtered_popval, x='property_value', bins=bins)
make_hist_kde(filtered_popval, x='tract_population', bins=bins)
make_hist_kde(filtered_popval, x='tract_minority_population_percent', bins=bins)
make_hist_kde(filtered_popval, x='ffiec_msa_md_median_family_income', bins=bins)
make_hist_kde(filtered_popval, x='tract_to_msa_income_percentage', bins=bins)
make_hist_kde(filtered_popval, x='tract_owner_occupied_units', bins=bins)
make_hist_kde(filtered_popval, x='tract_one_to_four_family_homes', bins=bins)
make_hist_kde(filtered_popval, x='tract_median_age_of_housing_units', bins=bins)

#%%

# make_scatter(filtered_popval, x='tract_population', y='property_value')
# make_scatter(filtered_popval, x='tract_minority_population_percent', y='property_value')
# make_scatter(filtered_popval, x='ffiec_msa_md_median_family_income', y='property_value')
# make_scatter(filtered_popval, x='tract_to_msa_income_percentage', y='property_value')
# make_scatter(filtered_popval, x='tract_owner_occupied_units', y='property_value')
# make_scatter(filtered_popval, x='tract_one_to_four_family_homes', y='property_value')
# make_scatter(filtered_popval, x='tract_median_age_of_housing_units', y='property_value')

#Pair Grid of Scatterplots
x_vars1=["tract_population", "tract_minority_population_percent", "tract_to_msa_income_percentage"]
x_vars2=["tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]

g1 = sns.PairGrid(filtered_popval, y_vars=["property_value"], x_vars=x_vars1, height=5)
g1.map(sns.regplot, color="#b3b3b3", scatter_kws={'alpha':0.03}, line_kws={"color":"#e5c494"})
plt.ticklabel_format(style='plain')
g2 = sns.PairGrid(filtered_popval, y_vars=["property_value"], x_vars=x_vars2, height=5)
g2.map(sns.regplot, color="#b3b3b3", scatter_kws={'alpha':0.03}, line_kws={"color":"#e5c494"})
plt.ticklabel_format(style='plain')
plt.show()

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%
#Check VIF
#resource: https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/

# the independent variables set
vars1=[7,9]
vars2=[7,9,10,11,12]

vif_filter = filtered_popval.iloc[:, np.r_[vars2]]
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = vif_filter.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(vif_filter.values, i)
                   for i in range(len(vif_filter.columns))]

#Display results
display(vif_data)

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%
#Model with all values chosen
model_test=ols(formula='property_value ~ C(derived_dwelling_category) + C(occupancy_type) + C(construction_method) + C(total_units) + tract_population + tract_minority_population_percent + ffiec_msa_md_median_family_income + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())


#%%
#Model based on VIF
model_vif=ols(data=filtered_popval, formula='property_value ~ tract_minority_population_percent + tract_to_msa_income_percentage')
model_vif_fit=model_vif.fit()
print(model_vif_fit.summary())

#%%
#Suspected important values

#derived_dwelling_category + occupancy_type + construction_method + total_units + interest_rate
#+ tract_population + tract_minority_population_percent + ffiec_msa_md_median_family_income + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units'

#Build model
model_test=ols(formula='property_value ~ C(occupancy_type) + C(total_units) + tract_minority_population_percent + tract_to_msa_income_percentage + tract_one_to_four_family_homes', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())

#%%
#Model all vars that showed low p-value (final?)
#+ C(total_units)
model_test=ols(formula='property_value ~ C(occupancy_type) + tract_minority_population_percent + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
# %%
