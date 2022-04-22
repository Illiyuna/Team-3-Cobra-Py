#%%

#############
## Imports ##
#############

#from cgi import test
#from operator import index
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import re

import matplotlib.ticker as ticker
import csv
import os 

from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor



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
#pd.set_option('display.float_format', lambda x: '%.2f' % x)

#Possible Graph Color Schemes
#color_list=['BuPu', 'PRGn', 'Pastel1', 'Pastel2', 'Set2', 'binary', 'bone', 'bwr',
#                 'bwr_r', 'coolwarm', 'cubehelix', 'gist_earth', 'gist_gray', 'icefire',]

# for color in color_list:
#     print(color)
#     g = sns.FacetGrid(df_all, col="world", hue='world', palette=color)
#     g.map_dataframe(sns.histplot, x="income00")
#     plt.show()

#Select Color Palette
sns.set_palette('Set2')

#RGB values of pallette
#print(sns.color_palette('Set2').as_hex())
#col_pallette=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

###############
## Functions ##
###############

def clean_strnumbers_to_numeric(val):
    
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
    
    #Note:
    # This is working, but values end up being float in df.
    # That is what is desired, but only converted to int here.
    # It's working, so I am leaving it alone.

##################################################

#CAN DELETE
# def clean_propval2(val):
    
#     if val == 'Exempt':
#         return 0        
#     else:
#         return float(val)

##################################################

def clean_int_rate(val):
    
    try:
        #remove any whitespace
        new_val=val.strip()
        #Attempt to convert directly to float
        new_val=float(val)
        #return NaN if negative
        return (new_val if new_val>=0 else np.nan)
    except:
        pass
    
    #Convert interest rate to 0, if exempt
    if val=='Exempt':
        return 0
    
    #Catch all for situations not covered
    return np.nan

    #Note:
    # Currently, Exempt is being turned into int_rate of 0
    # Is this desited? Other conversion better?

##################################################

def make_scatter(data, x, y):
    ax1=sns.regplot(data=data, x=x, y=y, color="#b3b3b3", scatter_kws={'alpha':0.03})
    ax1.lines[0].set_color("#e5c494")
    plt.ticklabel_format(style='plain')
    plt.xticks(rotation=45)
    plt.show()
    
    return None

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

    return None

##################################################

def create_vif_table(df, vars):

    vif_filter = df.iloc[:, np.r_[vars]]
  
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = vif_filter.columns
    
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(vif_filter.values, i)
                    for i in range(len(vif_filter.columns))]

    #Display results
    display(vif_data)
    
    return None

##################################################

    #************************#
##### NEW FUNCTIONS ADD HERE #####
    #************************#


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

###############
## Load Data ##
###############

#Load in full dc_df
filepath="data/dc_df.csv"
dc_df=pd.read_csv(filepath)

#%%

#show number of obs and display full dc_df head
print('\nShow head and number of observarions in FULL DC data set...\n')
print(f'Dc observations: {len(dc_df)}')
display(dc_df.head().style.set_sticky(axis="index"))

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

################
## Clean Data ##
################

#Convert columns to desired data types

#property value into float
dc_df['property_value']=dc_df.loc[:, 'property_value'].apply(clean_strnumbers_to_numeric)
#interest_rate into float
dc_df['interest_rate']=dc_df.loc[:, 'interest_rate'].apply(clean_int_rate)
#total_loan_costs into float
dc_df['total_loan_costs']=dc_df.loc[:, 'total_loan_costs'].apply(clean_strnumbers_to_numeric)

##################################################

#%%

#Filter Top-Level Data


    #**************************#
##### ADD HDMA Filters HERE ?? #####
    #**************************#

##################################################

#Check and filter for outliers ?? --using IQR*1.5
#Or just eyeball what looks/feels good ?? -- Current approach

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

#################
## Select Data ##
#################

#Column lists for subsets

#Initial variables of interest
intial_col_list =['activity_year', 
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
                'loan_term',  
                'multifamily_affordable_units', 
                'income', 
                'debt_to_income_ratio',
                'applicant_age', 
                'co_applicant_age', 
                'property_value',
                'derived_dwelling_category',
                'occupancy_type',
                "construction_method",
                'total_units',
                'interest_rate',
                'tract_population',
                'tract_minority_population_percent',
                'ffiec_msa_md_median_family_income',
                'tract_to_msa_income_percentage',
                'tract_owner_occupied_units',
                'tract_one_to_four_family_homes',
                'tract_median_age_of_housing_units']

over_charged_col_list=['total_loan_costs', 'interest_rate', 'derived_sex', 'derived_race']

propval_col_list=['property_value',
                'derived_dwelling_category',
                'occupancy_type',
                "construction_method",
                'total_units',
                'interest_rate',
                'tract_population',
                'tract_minority_population_percent',
                'ffiec_msa_md_median_family_income',
                'tract_to_msa_income_percentage',
                'tract_owner_occupied_units',
                'tract_one_to_four_family_homes',
                'tract_median_age_of_housing_units']

##################################################

#%%

#Create Subsets
init_df=dc_df[intial_col_list].copy()
overcharged_df=dc_df[over_charged_col_list].copy()
propval_df=dc_df[propval_col_list].copy()

##################################################

#%%

#Filter Subsets

#init_df filters
#or only filter for hdma settings here ??

#overcharged_df filters
#Set cost of loan cut off value
loan_cutoff=20000
#Set min and max interest rate values
intr_low1=1.7
intr_high1=4.8
intr_low2=1.5
intr_high2=5.5
#Set up filters for each and combined filter
loan_cond=overcharged_df['total_loan_costs']<loan_cutoff
intr_cond=((overcharged_df['interest_rate']>intr_low1) & (overcharged_df['interest_rate']<intr_high1))
intr_cond2=((overcharged_df['interest_rate']>intr_low2) & (overcharged_df['interest_rate']<intr_high2))
combined_cond=((overcharged_df['interest_rate']>intr_low2) & (overcharged_df['interest_rate']<intr_high2) & (overcharged_df['total_loan_costs']<loan_cutoff))

#propval_df filters
#Set cut off for value of property
propval_cutoff=2000000
#Set up filter
propval_cond=propval_df['property_value']<=propval_cutoff

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

########################################
## Summary Statistics After Filtering ##
########################################

#use to fix rows for df with many columns
#.style.set_sticky(axis="index"))

#Top-level, initial df
print('\nDisplay Summary statistics for Initial DF...\n')

    #*******************************#
##### INSERT Illiyuna's tables here #####
    #*******************************#

#%%

#overcharged subset
print('\nDisplay Summary statistics for Overcharged DF...\n')

#overcharged_df=overcharged_df.dropna()
#overcharged_df[loan_cond].describe()
#overcharged_df[intr_cond].describe()
#overcharged_df[intr_cond2].describe()
#overcharged_df[combined_cond].describe()

#%%

#propval subset
print('\nDisplay Summary statistics for Propval DF...\n')

#filter for conditions
filtered_popval=propval_df[propval_cond].copy() #don't need copy() if not making changes, but good habit
#Summary Statistics
print('Proval DF head')
display(filtered_popval.head().style.set_sticky(axis="index"))
print('\nPropval Numerical Variables, IQR Ranges')
display(filtered_popval.describe().style.set_sticky(axis="index"))
#Need to create create df of categorical variables and their value_count frequencies
#Then display as the categorical variable IQR range
#Below are the value counts
print('\nDistribution of dwelling category')
display(filtered_popval.derived_dwelling_category.value_counts())
print('\nDistribution of occupancy type')
display(filtered_popval.occupancy_type.value_counts())
print('\nDistribution of construction method')
display(filtered_popval.construction_method.value_counts())
print('\nDistribution of total units')
display(filtered_popval.total_units.value_counts())
#Corrplot
print('\nCorrelation Matrix of Propval Numerical Variables')
display(round(filtered_popval.corr(), 3).style.set_sticky(axis="index"))

#pairplot
#sns.pairplot(filtered_popval)

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

######################################
## Graphical Exploration: Inital DF ##
######################################

print('\nGraphical Exploration: Initial DF...\n')

    #*******************************#
##### INSERT Illiyuna's graphs here #####
    #*******************************#

#Graph Groupings of preapproval with action taken
#Need tables Illiyuna made for this to run
# data=pd.DataFrame(grouped.size()).reset_index()
# sns.barplot(data=data[data['preapproval']==1], x='action_taken', y=0)
# plt.title('Distribution of Actions Taken for Applicants who Were not Preapproved')
# plt.show()
# sns.barplot(data=data[data['preapproval']==2], x='action_taken', y=0)
# plt.title('Distribution of Actions Taken for Preapproved Applicants')
# plt.show()

##################################################

#%%

###########################################
## Graphical Exploration: Overcharged DF ##
###########################################

print('\nGraphical Exploration: Overcharged DF...\n')

overcharged_df=overcharged_df.dropna()

#Top-level graphs
print('\nOvercharged DF Top-level Graphs')

#barplot of race
print('\nBar plot of Race')
labs=['White', 'Race Not Available', 'Black or African American',
     'Asian', 'Joint', '2 or more minority races',
     'American Indian or Alaska Native',
     'Native Hawaiian or Other Pacific Islander', 'Other']
race=overcharged_df['derived_race'].value_counts()
g=sns.barplot(x=race.index, y=race.values,
              hue=race.index, hue_order=labs,
              dodge=False)
plt.xticks([])
plt.show()

#barplot of sex
print('\nBar plot of Sex')
sex=overcharged_df['derived_sex'].value_counts()
g=sns.barplot(x=sex.index, y=sex.values)
plt.show()

#total loan cost hist
print('\nTotal Loan Cost Distribution')
sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', bins=30)
plt.show()

#interest rates hist
print('\nInterest Rate Distribution')
sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', bins=25)
plt.xticks([])
plt.show()

#regplot total interest rate against total loan cost
print('\nScatterplot with LR line, Interest Rate Against Total Loan Cost')
g=sns.regplot(data=overcharged_df[combined_cond], x='total_loan_costs',
                y='interest_rate', scatter_kws={'alpha':0.03}, color="gray")
g.lines[0].set_color("pink")
plt.show()

##################################################

#%%

#Graphs grouping by interactive terms: Interest Rate
print('\nGraphs grouping by interactive terms: Interest Rate\n')

#interest rate by sex hist
print('\nInterest Rate by Sex, stacked')
sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', hue='derived_sex', multiple="stack", bins=25)
plt.xticks([])
plt.show()

print('\nInterest Rate by Sex, relative rate')
sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', hue='derived_sex', bins=25)
plt.xticks([])
plt.show()

#interest rate by race hist
print('\nInterest Rate by Race, stacked')
labs=['Native Hawaiian or Other Pacific Islander',
      'American Indian or Alaska Native', 'Joint',
      '2 or more minority races', 'Asian',
      'Black or African American', 'White',
      'Race Not Available']
sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', hue='derived_race', multiple="stack", bins=25)
             #hue_order=labs)
plt.xticks([])
plt.legend(title='Race', labels=labs, loc='center', bbox_to_anchor=(0.97, 0.75), borderaxespad=0.2)
plt.show()

print('\nInterest Rate by Race, relative rate')
sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', hue='derived_race', bins=25)#,
             #hue_order=labs)
plt.xticks([])
plt.legend(title='Race', labels=labs, loc='center', bbox_to_anchor=(0.97, 0.75), borderaxespad=0.2)
plt.show()

#Facet Grid of interest rate by sex
print('\nInterest Rate by Sex, Facet Graph')
g=sns.FacetGrid(overcharged_df[intr_cond], col="derived_sex", hue="derived_sex")
g.map_dataframe(sns.histplot, x="interest_rate", bins=25)
plt.show()

#Facet Grid of interest rate by race
print('\nInterest Rate by Race, Facet Graph')
g=sns.FacetGrid(overcharged_df[intr_cond], col="derived_race", hue="derived_race")
g.map_dataframe(sns.histplot, x="interest_rate", bins=25)
plt.show()


##################################################

#%%

#Graphs grouping by interactive terms: Total Loan Cost
print('\nGraphs grouping by interactive terms: Total Loan Cost\n')

#total loan costs by sex hist
print('\nTotal Loan Cost by Sex, stacked')
sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', hue='derived_sex', multiple="stack", bins=25)
plt.xticks([])
plt.show()

print('\nTotal Loan Cost by Sex, relative rate')
sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', hue='derived_sex', bins=25)
plt.xticks([])
plt.show()

#total loan costs by race hist
#LABS ARE WRONG ORDER! NEED TO CHECK!!
print('\nTotal Loan Cost by Race, stacked')
labs=['Native Hawaiian or Other Pacific Islander',
      'American Indian or Alaska Native', 'Joint',
      '2 or more minority races', 'Asian',
      'Black or African American', 'White',
      'Race Not Available']
sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', hue='derived_race', multiple="stack", bins=25)
             #hue_order=labs)
plt.xticks([])
plt.legend(title='Race', labels=labs, loc='center', bbox_to_anchor=(0.97, 0.75), borderaxespad=0.2)
plt.show()

print('\nTotal Loan Cost by Race, relative rate')
sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', hue='derived_race', bins=25)#,
             #hue_order=labs)
plt.xticks([])
plt.legend(title='Race', labels=labs, loc='center', bbox_to_anchor=(0.97, 0.75), borderaxespad=0.2)
plt.show()

#Facet Grid of total loan cost by sex
print('\nTotal Loan Cost by Sex, Facet Graph')
g=sns.FacetGrid(overcharged_df[loan_cond], col="derived_sex", hue="derived_sex")
g.map_dataframe(sns.histplot, x="total_loan_costs", bins=25)
plt.show()

#Facet Grid of total loan cost by race
print('\nTotal Loan Cost by Race, Facet Graph')
g=sns.FacetGrid(overcharged_df[loan_cond], col="derived_race", hue="derived_race")
g.map_dataframe(sns.histplot, x="total_loan_costs", bins=25)
plt.show()

##################################################

#%%

#violin plots for comparisons
print('\nViolin Plots Comparing Effects of Sex and Race on Total Loan Costs and Interest Rate...\n')

#sex total loan cost
print('\nDistribution of Total Loan Costs by Sex')
sns.violinplot(x="derived_sex",
             y="total_loan_costs",
             #hue="derived_race",
             data = overcharged_df[loan_cond])
plt.show()

#race total loan cost
print('\nDistribution of Total Loan Costs by Race')
sns.violinplot(x="derived_race",
             y="total_loan_costs",
             #hue="derived_race",
             data = overcharged_df[loan_cond])
plt.xticks(rotation=45, size='small')
plt.show()

#sex interest rate
print('\nDistribution of Interest Rates by Sex')
sns.violinplot(x="derived_sex",
             y="interest_rate",
             #hue="derived_race",
             data = overcharged_df[loan_cond])
plt.show()

#race interest rate
print('\nDistribution of Interest Rates by Race')
sns.violinplot(x="derived_race",
             y="interest_rate",
             #hue="derived_race",
             data = overcharged_df[loan_cond])
plt.xticks(rotation=45, size='small')
plt.show()

#%%

#######################################
## Graphical Exploration: Propval DF ##
#######################################

print('\nGraphical Exploration: Propval DF...\n')

#Bar graphs

#%%
#Box plots

#%%
#hist plots of all variables examined in this question
print('\nDistributions of Community Factors')
#binw=None
strg=16
rice=36
bins=strg
print('\nProperty Value')
make_hist_kde(filtered_popval, x='property_value', bins=bins)
print('\nTract Population')
make_hist_kde(filtered_popval, x='tract_population', bins=bins)
print('\nPercentage of Tract Population that is in Minority Group')
make_hist_kde(filtered_popval, x='tract_minority_population_percent', bins=bins)
print('\nRegional Median Family Income')
make_hist_kde(filtered_popval, x='ffiec_msa_md_median_family_income', bins=bins)
print('\nTract Median Family Income as Percentage of Regional Median Family Income')
make_hist_kde(filtered_popval, x='tract_to_msa_income_percentage', bins=bins)
print('\nNumber of Owner Occupied Units')
make_hist_kde(filtered_popval, x='tract_owner_occupied_units', bins=bins)
print('\nNumber of 1-4 Family Homes')
make_hist_kde(filtered_popval, x='tract_one_to_four_family_homes', bins=bins)
print('\nMedian Age of Units')
make_hist_kde(filtered_popval, x='tract_median_age_of_housing_units', bins=bins)

#%%

print('\nViewing Scatterplots with LR Line of Property Value Against Community Factors...\n')

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

#%%
#Additional variable comparison
print('\nScatterplots of Correlated Community Factors Against One Another...\n')

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

##########################
## Check VIF Propval DF ##
##########################

print('\nVIF Scores When Considering Different Community Factors')

#resource: https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/

filtered_popval=propval_df[propval_cond].copy()

#Check for collinearity with VIF
create_vif_table(filtered_popval, [6,7,9,10,11,12])
create_vif_table(filtered_popval, [7,9,10,11,12])
create_vif_table(filtered_popval, [7,9,10,11])
create_vif_table(filtered_popval, [7,9,11])
create_vif_table(filtered_popval, [7,9])

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

###############################
## OLS Models for Propval DF ##
###############################

print('\nModel Outcomes...\n')

#%%
#Model with all values chosen
print('\nModel with all values chosen\n')
model_test=ols(formula='property_value ~ C(derived_dwelling_category) + C(occupancy_type) + C(construction_method) + C(total_units) + tract_population + tract_minority_population_percent + ffiec_msa_md_median_family_income + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())

#%%
#Model based on VIF
print('\nModel for Factors With VIF Scores Less than 5\n')
model_vif=ols(data=filtered_popval, formula='property_value ~ tract_minority_population_percent + tract_to_msa_income_percentage')
model_vif_fit=model_vif.fit()
print(model_vif_fit.summary())

#%%
#Suspected important values
print('\nModel For Suspected Important Factors, With VIF less than 10\n')
#derived_dwelling_category + occupancy_type + construction_method + total_units + interest_rate
#+ tract_population + tract_minority_population_percent + ffiec_msa_md_median_family_income + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units'

#Build model
model_test=ols(formula='property_value ~ C(occupancy_type) + C(total_units) + tract_minority_population_percent + tract_to_msa_income_percentage + tract_one_to_four_family_homes', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())

#%%
#Model all vars that showed low p-value (final?)
print('\nModel For all Factors that had a low p-value in the All-Factors-Model\n')
#+ C(total_units)
model_test=ols(formula='property_value ~ C(occupancy_type) + tract_minority_population_percent + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
# %%
