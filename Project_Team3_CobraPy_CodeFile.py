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

def make_hist_kde(data, x, bins, hue):
    #change line color 
    #source: https://github.com/mwaskom/seaborn/issues/2344
    
    #Initialize graph
    ax1=sns.histplot(data=data, x=x, bins=bins, color="#8da0cb", hue=hue,
                     multiple="stack", kde=True, line_kws=dict(linewidth=4))
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

def create_count_df(df, col_name):
    
    #Get count of every level of variable
    val_count=df[col_name].value_counts()
    #Create DF of the counts
    count_df=pd.DataFrame({"categorical_levels": val_count.index,
                  "count": val_count.values})
    #Convert level counts into percentages of total
    #freq_df['freq']=round(freq_df['freq'].apply(lambda x: x/sum(freq_df['freq'])), 3)
    
    return count_df

##################################################

def create_freq_df(df, col_name):
    
    freq_df=create_count_df(df,col_name).rename(columns={'count':'freq'})
    freq_df['freq']=round(freq_df['freq'].apply(lambda x: x/sum(freq_df['freq'])), 3)
    
    return freq_df

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
#filepath="data/dc_df.csv"
dc_df=pd.read_csv('data/dc_df.csv')

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
# discount_points into float
dc_df['discount_points']=dc_df.loc[:, 'discount_points'].apply(clean_strnumbers_to_numeric)
# lender-credits into float 
dc_df['lender_credits']=dc_df.loc[:, 'lender_credits'].apply(clean_strnumbers_to_numeric)

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

#################
## Select Data ##
#################

#Column lists for subsets

#Overcharged DF
over_charged_col_list=['total_loan_costs', 
                       'interest_rate', 
                       'derived_sex', 
                       'derived_race',
                       'discount_points',
                       'lender_credits',
                       'loan_type', #added
                       'loan_amount'] #added

#Propety Value DF
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
overcharged_df=dc_df[over_charged_col_list].copy()
propval_df=dc_df[propval_col_list].copy()

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

######################
## EDA: OVERCHARGED ##
######################

#overcharged_df filters

#Set cost of loan cut off value
loan_cutoff=20000
#Set min and max interest rate values
# min
intr_low1=1.7
intr_high1=4.8
# max 
intr_low2=1.5
intr_high2=5.5
#Set up filters for each and combined filter
loan_cond=overcharged_df['total_loan_costs']<loan_cutoff
intr_cond=((overcharged_df['interest_rate']>intr_low1) & (overcharged_df['interest_rate']<intr_high1))
intr_cond2=((overcharged_df['interest_rate']>intr_low2) & (overcharged_df['interest_rate']<intr_high2))
combined_cond=((overcharged_df['interest_rate']>intr_low2) & (overcharged_df['interest_rate']<intr_high2) & (overcharged_df['total_loan_costs']<loan_cutoff))

##################################################

#%%

#####################################
## Summary Statistics: OVERCHARGED ##
#####################################

#use to fix rows for df with many columns
#.style.set_sticky(axis="index"))

# Overcharged Subset

print('\nDisplay Summary statistics for Overcharged DF...\n')

overcharged_df=overcharged_df.dropna()
overcharged_df.isnull().sum() / overcharged_df.shape[0] * 100
# overcharged_df[loan_cond].describe()
# overcharged_df[intr_cond].describe()
# overcharged_df[intr_cond2].describe()
# overcharged_df[combined_cond].describe()

# Dropping "Sex Not Available" From Derived Sex
overcharged_df = overcharged_df[overcharged_df.derived_sex != 'Sex Not Available']
overcharged_df = overcharged_df[overcharged_df.derived_race != 'Race Not Available']

#*********************************#
##### DUPLICATE NEEDS DELETED #####
#*********************************#

# # Overcharged subset
# print('\nDisplay Summary statistics for Overcharged DF...\n')

# overcharged_df=overcharged_df.dropna()
# overcharged_df.isnull().sum() / overcharged_df.shape[0] * 100
# # overcharged_df[loan_cond].describe()
# # overcharged_df[intr_cond].describe()
# # overcharged_df[intr_cond2].describe()
# # overcharged_df[combined_cond].describe()

# # Dropping "Sex Not Available" From Derived Sex
# overcharged_df = overcharged_df[overcharged_df.derived_sex != 'Sex Not Available']
# overcharged_df = overcharged_df[overcharged_df.derived_race!= 'Race Not Available']

# %%
# First glimpse of variable characteristics grouped by sex or race

print(overcharged_df.groupby('derived_sex').mean())
print(overcharged_df.groupby('derived_race').mean())

# From the mean tables,
# Single women paid lower loan costs compared to men and joint on average.
# However, this makes sense since they receive higher lender credits than men or married couples.
# Single women pay highest interest and get less discount points on average. 
# Men get more discount points (IR reduction)

# Asians, Blacks and Hawaiians pays most loan costs.
# This is weird because Black people receive the highes lender credits. 
# White people have lowest LC still lower TLC. 

# Black people and Hawaiians have highest IR, but Hawaiians receive the most discount points. 

# Are finacial inst. charging higher IR despite discounts???
### What is the probability of getting a high cost loan based on your sex & race??
### Are men getting more or bigger loans than women? 
### Blacks get more credits but still get higher IR? Hypocrisy of financial inst.


#%%

###########################################
## Graphical Exploration: Overcharged DF ##
###########################################

print('\nGraphical Exploration: Overcharged DF...\n')

#Top-level graphs
print('\nOvercharged DF Top-level Graphs')

# Barplot of Race
print('\nBar plot of Race')
labs=['White','Race Not Available', 'Black or African American',
     'Asian', 'Joint', '2 or more minority races',
     'American Indian or Alaska Native',
     'Native Hawaiian or Other Pacific Islander', 'Other']
race=overcharged_df['derived_race'].value_counts()
g=sns.barplot(x=race.index, y=race.values,
              hue=race.index, hue_order=labs,
              dodge=False)
plt.title('Barplot of Race')
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

# discount points hist
print('\nDiscount Points Distribution')
sns.histplot(data=overcharged_df, x='discount_points', bins=50)
plt.show()

# lender credits hist
print('\nLender Credits Distribution')
sns.histplot(data=overcharged_df, x='lender_credits', bins=50)
plt.show()

##################################################

#%%
# Overcharged Gender Plots 

# Hypo: is that single women are overcharged compared to single men. 

sns.stripplot(data=overcharged_df,
              x='derived_sex', 
              y='interest_rate',
              palette='mako')
plt.xlabel('Derived Sex',size=14)
plt.ylabel('Interest Rate',size=14)
plt.title('Interest Rates by Sex') 
plt.show()

sns.stripplot(data=overcharged_df, 
                x='derived_sex',
                y='total_loan_costs')
plt.xlabel('Derived Sex',size=14)
plt.ylabel('Total Loan Cost',size=14)
plt.title('Total Loan Costs by Sex')

sns.barplot(data=overcharged_df, 
                x='derived_sex',
                y='discount_points')
plt.xlabel('Derived Sex',size=14)
plt.ylabel('Discount Points',size=14)
plt.title('Discount Points by Sex', size=14)

sns.barplot(data=overcharged_df, 
                x='derived_sex',
                y='lender_credits')
plt.xlabel('Derived Sex',size=14)
plt.ylabel('Lender Credits',size=14)
plt.title('Lender Credits by Sex', size=14)

# Overcharged Race Plots 

# Hypo: minorities are charged more than caucasians  

sns.stripplot(data=overcharged_df,
              x='derived_race', 
              y='interest_rate',
              palette='mako')
plt.xlabel('Derived Race',size=14)
plt.ylabel('Interest Rate',size=14)
plt.title('Interest Rates by Race') 
plt.xticks(rotation=45)
plt.show()

sns.stripplot(data=overcharged_df, 
                x='derived_race',
                y='total_loan_costs')
plt.xlabel('Derived Race',size=14)
plt.ylabel('Total Loan Cost',size=14)
plt.title('Total Loan Costs by Race') 
plt.xticks(rotation=45)
plt.show()

# Fix x-ticks for race plots. 

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
# Correlation Matrix for Overcharged 

corrMatrix = overcharged_df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# %%
# SEX

#Facet Grid of total loan cost by sex
print('\nTotal Loan Cost by Sex, Facet Graph')
g=sns.FacetGrid(overcharged_df[loan_cond], col="derived_sex", hue="derived_sex")
g.map_dataframe(sns.histplot, x="total_loan_costs", bins=25)
plt.show()

#Facet Grid of interest rate by sex
print('\nInterest Rate by Sex, Facet Graph')
g=sns.FacetGrid(overcharged_df[intr_cond], col="derived_sex", hue="derived_sex")
g.map_dataframe(sns.histplot, x="interest_rate", bins=25)
plt.show()

#Facet Grid of discount points by sex
print('\nDiscount Points by Sex, Facet Graph')
g=sns.FacetGrid(overcharged_df, col="derived_sex", hue="derived_sex")
g.map_dataframe(sns.histplot, x="discount_points", bins=25)
plt.show()

#Facet Grid of lender credits by sex
print('\nLender Credits by Sex, Facet Graph')
g=sns.FacetGrid(overcharged_df, col="derived_sex", hue="derived_sex")
g.map_dataframe(sns.histplot, x="lender_credits", bins=25)
plt.show()


#%%
# RACE

#Facet Grid of total loan cost by race
print('\nTotal Loan Cost by Race, Facet Graph')
g=sns.FacetGrid(overcharged_df[loan_cond], col="derived_race", hue="derived_race")
g.map_dataframe(sns.histplot, x="total_loan_costs", bins=25)
plt.show()

#Facet Grid of interest rate by race
print('\nInterest Rate by Race, Facet Graph')
g=sns.FacetGrid(overcharged_df[intr_cond], col="derived_race", hue="derived_race")
g.map_dataframe(sns.histplot, x="interest_rate", bins=25)
plt.show()

#Facet Grid of discount points by race
print('\nDiscount Points by Race, Facet Graph')
g=sns.FacetGrid(overcharged_df, col="derived_race", hue="derived_race")
g.map_dataframe(sns.histplot, x="discount_points", bins=25)
plt.show()

#Facet Grid of lender credits by race
print('\nLender Credits by Race, Facet Graph')
g=sns.FacetGrid(overcharged_df, col="derived_race", hue="derived_race")
g.map_dataframe(sns.histplot, x="lender_credits", bins=25)
plt.show()


#%%

# Are men getting more/ bigger loans than women?? 

sns.barplot(data=overcharged_df, 
            x='loan_type', 
            y='loan_amount', 
            hue='derived_sex', ci=None)
plt.xlabel('Type of Loan',size=14)
plt.ylabel('Average Loan Amount',size=14)
plt.title('Average Loan Amount & Type by Sex') 
x_var= ['Conventional', 
        'FHA Insured', 
        'VA Insured',
        'RHS or FSA Insured']
plt.xticks([0,1,2,3], x_var, rotation=20)       
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()

# Yes men are getting higher loans on average, esp for conventional and FHA loans.
# Veterans are not showing much difference. 



###########################################
## Graphical Exploration: Overcharged DF ## Andy's work 
###########################################

# If a graph is useful, use, otherwise can be deleted

# print('\nGraphical Exploration: Overcharged DF...\n')

# overcharged_df=overcharged_df.dropna()

# #Top-level graphs
# print('\nOvercharged DF Top-level Graphs')

# #barplot of race
# print('\nBar plot of Race')
# labs=['White', 'Race Not Available', 'Black or African American',
#      'Asian', 'Joint', '2 or more minority races',
#      'American Indian or Alaska Native',
#      'Native Hawaiian or Other Pacific Islander', 'Other']
# race=overcharged_df['derived_race'].value_counts()
# g=sns.barplot(x=race.index, y=race.values,
#               hue=race.index, hue_order=labs,
#               dodge=False)
# plt.xticks([])
# plt.show()

# #barplot of sex
# print('\nBar plot of Sex')
# sex=overcharged_df['derived_sex'].value_counts()
# g=sns.barplot(x=sex.index, y=sex.values)
# plt.show()

# #total loan cost hist
# print('\nTotal Loan Cost Distribution')
# sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', bins=30)
# plt.show()

# #interest rates hist
# print('\nInterest Rate Distribution')
# sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', bins=25)
# plt.xticks([])
# plt.show()

# # discount points hist
# print('\nDiscount Points Distribution')
# sns.histplot(data=overcharged_df, x='discount_points', bins=50)
# plt.show()

# # lender credits hist
# print('\nLender Credits Distribution')
# sns.histplot(data=overcharged_df, x='lender_credits', bins=50)
# plt.show()

# #regplot interest rate against total loan cost
# print('\nScatterplot with LR line, Interest Rate Against Total Loan Cost')
# g=sns.regplot(data=overcharged_df[combined_cond], x='total_loan_costs',
#                 y='interest_rate', scatter_kws={'alpha':0.03}, color="gray")
# g.lines[0].set_color("pink")
# plt.show()

# #regplot discount points vs lender credits 
# print('\nScatterplot with LR line, Discount Points Against Lender Credits')
# g=sns.regplot(data=overcharged_df, x='lender_credits',
#                 y='discount_points', scatter_kws={'alpha':0.03}, color="gray")
# g.lines[0].set_color("pink")
# plt.show()

# #regplot discount points vs interest rate 
# print('\nScatterplot with LR line, Interest Rate Against Discount Points')
# g=sns.regplot(data=overcharged_df, x='discount_points',
#                 y='interest_rate', scatter_kws={'alpha':0.03}, color="gray")
# g.lines[0].set_color("pink")
# plt.show()

# #regplot total loan costs vs lender credits 
# print('\nScatterplot with LR line, Total Loan Costs Against Lender Credits')
# g=sns.regplot(data=overcharged_df, x='lender_credits',
#                 y='total_loan_costs', scatter_kws={'alpha':0.03}, color="gray")
# g.lines[0].set_color("pink")
# plt.show()

# #regplot interest rate vs lender credits 
# print('\nScatterplot with LR line, Interest Rates Against Lender Credits')
# g=sns.regplot(data=overcharged_df, x='lender_credits',
#                 y='interest_rate', scatter_kws={'alpha':0.03}, color="gray")
# g.lines[0].set_color("pink")
# plt.show()

# ##################################################

# #%%

# #Graphs grouping by interactive terms: Interest Rate
# print('\nGraphs grouping by interactive terms: Interest Rate\n')

# #interest rate by sex hist
# print('\nInterest Rate by Sex, stacked')
# sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', hue='derived_sex', multiple="stack", bins=25)
# plt.xticks([])
# plt.show()

# print('\nInterest Rate by Sex, relative rate')
# sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', hue='derived_sex', bins=25)
# plt.xticks([])
# plt.show()

# #interest rate by race hist
# print('\nInterest Rate by Race, stacked')
# labs=['Native Hawaiian or Other Pacific Islander',
#       'American Indian or Alaska Native', 'Joint',
#       '2 or more minority races', 'Asian',
#       'Black or African American', 'White',
#       'Race Not Available']
# sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', hue='derived_race', multiple="stack", bins=25)
#              #hue_order=labs)
# plt.xticks([])
# plt.legend(title='Race', labels=labs, loc='center', bbox_to_anchor=(0.97, 0.75), borderaxespad=0.2)
# plt.show()

# print('\nInterest Rate by Race, relative rate')
# sns.histplot(data=overcharged_df[intr_cond], x='interest_rate', hue='derived_race', bins=25)#,
#              #hue_order=labs)
# plt.xticks([])
# plt.legend(title='Race', labels=labs, loc='center', bbox_to_anchor=(0.97, 0.75), borderaxespad=0.2)
# plt.show()

# #Facet Grid of interest rate by sex
# print('\nInterest Rate by Sex, Facet Graph')
# g=sns.FacetGrid(overcharged_df[intr_cond], col="derived_sex", hue="derived_sex")
# g.map_dataframe(sns.histplot, x="interest_rate", bins=25)
# plt.show()

# #Facet Grid of interest rate by race
# print('\nInterest Rate by Race, Facet Graph')
# g=sns.FacetGrid(overcharged_df[intr_cond], col="derived_race", hue="derived_race")
# g.map_dataframe(sns.histplot, x="interest_rate", bins=25)
# plt.show()


# ##################################################

# #%%

# #Graphs grouping by interactive terms: Total Loan Cost
# print('\nGraphs grouping by interactive terms: Total Loan Cost\n')

# #total loan costs by sex hist
# print('\nTotal Loan Cost by Sex, stacked')
# sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', hue='derived_sex', multiple="stack", bins=25)
# plt.xticks([])
# plt.show()

# print('\nTotal Loan Cost by Sex, relative rate')
# sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', hue='derived_sex', bins=25)
# plt.xticks([])
# plt.show()

# #total loan costs by race hist
# #LABS ARE WRONG ORDER! NEED TO CHECK!!
# print('\nTotal Loan Cost by Race, stacked')
# labs=['Native Hawaiian or Other Pacific Islander',
#       'American Indian or Alaska Native', 'Joint',
#       '2 or more minority races', 'Asian',
#       'Black or African American', 'White',
#       'Race Not Available']
# sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', hue='derived_race', multiple="stack", bins=25)
#              #hue_order=labs)
# plt.xticks([])
# plt.legend(title='Race', labels=labs, loc='center', bbox_to_anchor=(0.97, 0.75), borderaxespad=0.2)
# plt.show()

# print('\nTotal Loan Cost by Race, relative rate')
# sns.histplot(data=overcharged_df[loan_cond], x='total_loan_costs', hue='derived_race', bins=25)#,
#              #hue_order=labs)
# plt.xticks([])
# plt.legend(title='Race', labels=labs, loc='center', bbox_to_anchor=(0.97, 0.75), borderaxespad=0.2)
# plt.show()

# #Facet Grid of total loan cost by sex
# print('\nTotal Loan Cost by Sex, Facet Graph')
# g=sns.FacetGrid(overcharged_df[loan_cond], col="derived_sex", hue="derived_sex")
# g.map_dataframe(sns.histplot, x="total_loan_costs", bins=25)
# plt.show()

# #Facet Grid of total loan cost by race
# print('\nTotal Loan Cost by Race, Facet Graph')
# g=sns.FacetGrid(overcharged_df[loan_cond], col="derived_race", hue="derived_race")
# g.map_dataframe(sns.histplot, x="total_loan_costs", bins=25)
# plt.show()

# ##################################################

# #%%

# #violin plots for comparisons
# print('\nViolin Plots Comparing Effects of Sex and Race on Total Loan Costs and Interest Rate...\n')

# #sex total loan cost
# print('\nDistribution of Total Loan Costs by Sex')
# sns.violinplot(x="derived_sex",
#              y="total_loan_costs",
#              #hue="derived_race",
#              data = overcharged_df[loan_cond])
# plt.show()

# #race total loan cost
# print('\nDistribution of Total Loan Costs by Race')
# sns.violinplot(x="derived_race",
#              y="total_loan_costs",
#              #hue="derived_race",
#              data = overcharged_df[loan_cond])
# plt.xticks(rotation=45, size='small')
# plt.show()

# #sex interest rate
# print('\nDistribution of Interest Rates by Sex')
# sns.violinplot(x="derived_sex",
#              y="interest_rate",
#              #hue="derived_race",
#              data = overcharged_df[loan_cond])
# plt.show()

# #race interest rate
# print('\nDistribution of Interest Rates by Race')
# sns.violinplot(x="derived_race",
#              y="interest_rate",
#              #hue="derived_race",
#              data = overcharged_df[loan_cond])
# plt.xticks(rotation=45, size='small')
# plt.show()




#%%

#######################################################
## Statistical Tests and Model Building: OVERCHARGED ##
####################################################### 

overcharged_stats =overcharged_df.copy()
## Anova Tests for Sex
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#%%
# One way Anova Total loan cost vs Sex

# Total Loan Costs 
model_TLC= ols('total_loan_costs ~ C(derived_sex)',
            data=overcharged_stats).fit()
result_TLC = sm.stats.anova_lm(model_TLC, type=1)
  
# Print the result
print(result_TLC)

tukey_TLC = pairwise_tukeyhsd(endog=overcharged_stats['total_loan_costs'], groups=overcharged_stats['derived_sex'], alpha=0.05)
print(tukey_TLC)

# Females pay on average $1,832 LESS than "joint" borrowers.
# No statsitical significance for Female vs Male
# Males pay on average $1,033 LESS than "joint" borrowers.  

#%%
# Boxplot for Total Loan Costs

#overcharged_stats.boxplot(column = 'total_loan_costs', by = 'derived_sex', color = [1:3])

sns.boxplot(data = overcharged_stats, y = 'total_loan_costs', x = 'derived_sex', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
plt.title('Boxplot. Total Loan Costs vs Derived Sex (w/o Outliers)')
plt.xlabel('Sex')
plt.ylabel('Total Loan Costs ($)')


#%%
# Interest Rates 
model_IR= ols('interest_rate ~ C(derived_sex)',
            data=overcharged_stats).fit()
result_IR = sm.stats.anova_lm(model_IR, type=1)
  
# Print the result
print(result_IR)

tukey_IR = pairwise_tukeyhsd(endog=overcharged_stats['interest_rate'], groups=overcharged_stats['derived_sex'], alpha=0.05)
print(tukey_IR)

# Females pay 0.138 MORE points on their interest rate than joint borrowers.
# Females pay 0.86 MORE points on interest rates than Males.
# No statistical significant diff between joint and males

#%%
#Boxplot interet rates vs derived sex
sns.boxplot(data = overcharged_stats, y = 'interest_rate', x = 'derived_sex', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
plt.title('Boxplot. Interest Rates vs Derived Sex (w/o Outliers)')
plt.xlabel('Sex')
plt.ylabel('Interest rates (%)')


#%%
# Lender Credits
model_LC= ols('lender_credits ~ C(derived_sex)',
            data=overcharged_stats).fit()
result_LC = sm.stats.anova_lm(model_LC, type=1)
  
# Print the result
print(result_LC)

tukey_LC = pairwise_tukeyhsd(endog=overcharged_stats['lender_credits'], groups=overcharged_stats['derived_sex'], alpha=0.05)
print(tukey_LC)

# All groups are statistically different. This may be due finance perception, risk appetite and overall education, and not necesarily due to gender. 

#%%
#Boxplot Lender credits vs derived sex
sns.boxplot(data = overcharged_stats, y = 'lender_credits', x = 'derived_sex', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
plt.title('Boxplot. Lender Credits vs Derived Sex (w/o Outliers)')
plt.xlabel('Sex')
plt.ylabel('Lender Credits ($)')

#%%
# Discount Points 
model_DP= ols('discount_points ~ C(derived_sex)',
            data=overcharged_stats).fit()
result_DP = sm.stats.anova_lm(model_DP, type=1)
  
# Print the result
print(result_DP)

tukey_DP = pairwise_tukeyhsd(endog=overcharged_stats['discount_points'], groups=overcharged_stats['derived_sex'], alpha=0.05)
print(tukey_DP)

# All groups are statistically significantly different. This may be due finance perception, risk appetite and overall education, and not necesarily due to gender.

#%%
#Boxplot Discount Points vs derived sex
sns.boxplot(data = overcharged_stats, y = 'discount_points', x = 'derived_sex', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
plt.title('Boxplot. Discount Points vs Derived Sex (w/o Outliers)')
plt.xlabel('Sex')
plt.ylabel('Discount Points ($)')

#%%
# One way Anova for Race.

# Total Loan Costs 
model_TLC= ols('total_loan_costs ~ C(derived_race)',
            data=overcharged_stats).fit()
result_TLC = sm.stats.anova_lm(model_TLC, type=1)
  
# Print the result
print(result_TLC)

tukey_TLC = pairwise_tukeyhsd(endog=overcharged_stats['total_loan_costs'], groups=overcharged_stats['derived_race'], alpha=0.05)
print(tukey_TLC)

# Total loan costs are, on average, the same for every group. No statistical significantly differences found.

#%%
#Boxplot Total Loan Costs vs Derived Race
sns.boxplot(data = overcharged_stats, y = 'total_loan_costs', x = 'derived_race', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
plt.title('Boxplot. Total Loan Cost vs \n Derived Race (w/o Outliers)')
plt.xlabel('Race')
plt.ylabel('Total Loan Costs ($)') 

#%%
# Interest Rates vs Race
model_IR= ols('interest_rate ~ C(derived_race)',
            data=overcharged_stats).fit()
result_IR = sm.stats.anova_lm(model_IR, type=1)
  
# Print the result
print(result_IR)

tukey_IR = pairwise_tukeyhsd(endog=overcharged_stats['interest_rate'], groups=overcharged_stats['derived_race'], alpha=0.05)
print(tukey_IR)

# AfrAms pay, on average 22.2% MORE in Interest Rates than "joint" borrowers.
# AfrAms. pay, on average, 19.2% MORE in interest rates than their white counterparts.

#%%
#Boxplot Interest rates vs Derived Race
g1 = sns.boxplot(data = overcharged_stats, y = 'interest_rate', x = 'derived_race', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(xticklabels=[])
g1.set(title ='Boxplot. Interest Rate vs Derived Race (w/o Outliers)')
plt.legend(overcharged_stats['derived_race'].unique())
plt.xlabel('Race')
plt.ylabel('Interest rate (%)') 

#%%
# Lender Credits vs Race
model_LC= ols('lender_credits ~ C(derived_race)',
            data=overcharged_stats).fit()
result_LC = sm.stats.anova_lm(model_LC, type=1)
  
# Print the result
print(result_LC)

tukey_LC = pairwise_tukeyhsd(endog=overcharged_stats['lender_credits'], groups=overcharged_stats['derived_race'], alpha=0.05)
print(tukey_LC)

#  No statistical significance found amon the Race groups for lender credits.

#%%
# Discount Points vs Race
model_DP= ols('discount_points ~ C(derived_race)',
            data=overcharged_stats).fit()
result_DP = sm.stats.anova_lm(model_DP, type=1)
  
# Print the result
print(result_DP)

tukey_DP = pairwise_tukeyhsd(endog=overcharged_stats['discount_points'], groups=overcharged_stats['derived_race'], alpha=0.05)
print(tukey_DP)

# #  No statistical significance found amon the Race groups for discount points.
 
#%%
### TWO-WAY ANOVAs

# Total Loan Coats 
model = ols('total_loan_costs ~ C(derived_sex) + C(derived_race) +\
C(derived_sex):C(derived_race)',
            data=overcharged_stats).fit()
result = sm.stats.anova_lm(model, type=2)
print(result)

overcharged_stats['combination'] = overcharged_stats.derived_sex + overcharged_stats.derived_race

tukey_TLC2 = pairwise_tukeyhsd(endog=overcharged_stats['total_loan_costs'], groups=overcharged_stats['combination'] , alpha=0.05)

print(tukey_TLC2)
# Asian females, when taking a loan, pay on average $3,653 MORE than White "joint" borrowers.
# White females, when taking a loan, pay on average $3,653 MORE than White "joint" borrowers.
# No other statsitically significant difference was found in the interactions.


#%%
# Interest Rate
model = ols('interest_rate ~ C(derived_sex) + C(derived_race) +\
C(derived_sex):C(derived_race)',
            data=overcharged_stats).fit()
result = sm.stats.anova_lm(model, type=2)
  
print(result)

#overcharged_stats['combination'] = overcharged_stats.derived_sex + overcharged_stats.derived_race

tukey_IR2 = pairwise_tukeyhsd(endog=overcharged_stats['interest_rate'], groups=overcharged_stats['combination'] , alpha=0.05)

print(tukey_IR2)

# Female AfrAms pay on average 0.263 LESS on their interest rates than their White "joint" counterparts.
# Female AfrAms pay on average 0.221 LESS on their interest rates than their Male "joint" counterparts.
# White "joint" pay on average 0.185 MORE than their interest rates than their Male AfrAms counterparts.
 

#%%
# Lender Credit 
model = ols('lender_credits ~ C(derived_sex) + C(derived_race) +\
C(derived_sex):C(derived_race)',
            data=overcharged_stats).fit()
result = sm.stats.anova_lm(model, type=2)
  
print(result)

#overcharged_stats['combination'] = overcharged_stats.derived_sex + overcharged_stats.derived_race

tukey_LC2 = pairwise_tukeyhsd(endog=overcharged_stats['lender_credits'], groups=overcharged_stats['combination'] , alpha=0.05)

print(tukey_LC2)
# sex or race doesnt affect lender credits.

#%%
# Discount Points
model = ols('discount_points ~ C(derived_sex) + C(derived_race) +\
C(derived_sex):C(derived_race)',
            data=overcharged_stats).fit()
result = sm.stats.anova_lm(model, type=2)
  
print(result)

tukey_DP2 = pairwise_tukeyhsd(endog=overcharged_stats['discount_points'], groups=overcharged_stats['combination'] , alpha=0.05)

print(tukey_DP2)
# Asian females agree on average to an additional $2,264 on Discount points than their White "joint" counterparts.
# AfrAm females agree on average to an additional $1,267 on Discount points than their White "joint" counterparts.
# White females agree on average to an additional $1,214 on Discount points than their White "joint" counterparts.

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

##################
## EDA: Propval ##
##################

#propval_df filters

#Set cut off for value of property
propval_cutoff=3600000
#3.6 mil would be 1sd cutoff for filter
#Set up filter
propval_cond=propval_df['property_value']<=propval_cutoff
#filter for conditions
filtered_popval=propval_df[propval_cond].copy() #don't need copy() if not making changes, but good habit

#%%

#################################
## Summary Statistics: PROPVAL ##
#################################

print('\nDisplay Summary statistics for Propval DF...\n')

#Summary Statistics
print('Proval DF head')
display(filtered_popval.head().style.set_sticky(axis="index"))
print('\nPropval Numerical Variables, IQR Ranges')
display(filtered_popval.describe().style.set_sticky(axis="index"))

#COMMENTS
# Median Family Income -- only 1, since just looking at DMV region
#                      -- can drop

#%%

#Create create df for each categorical variables and their value_count frequencies

#Dwelling Type
print('\nDistribution of dwelling category')
dwelling_type_freq=create_freq_df(filtered_popval, 'derived_dwelling_category')
display(dwelling_type_freq)
#Occupancy Type
print('\nDistribution of occupancy type')
occupancy_type_freq=create_freq_df(filtered_popval, 'occupancy_type')
display(occupancy_type_freq)
#Construction Method
print('\nDistribution of construction method')
construction_method_freq=create_freq_df(filtered_popval, 'construction_method')
display(construction_method_freq)
#Total units
print('\nDistribution of total units')
total_units_freq=create_freq_df(filtered_popval, 'total_units')
display(total_units_freq)

#COMMENTS
#
# Only Occupancy Type appears to have any kind of breakdown amongst the levels 

#%%
#Corrplot
print('\nCorrelation Matrix of Propval Numerical Variables')
display(round(filtered_popval.corr(), 3).style.set_sticky(axis="index"))

corrMatrix_propval = filtered_popval.iloc[:, np.r_[0,6:12]].corr()
sns.heatmap(corrMatrix_propval, annot=True)
plt.show()

#Property Value is correlated with:
# -Tract perc of Regional Median Family Income (+)
# - Tract minority population percentage (-)
# - Owner occupied units (small +)
# - Number of 1-4 family homes (small +)

#Tract Population appears to be captured in
# -Number of onwer occupied units (+)
# -Number of 1-4 familt homes (+)

#Tract Minority Population Percentage also connected to
# -Number of Onwer occupied units (-)

#Tract perc of Regional Median Famliy Income also connect to
# -Number onwer occupied units (+)

#Number Onwer Occupied Units also connected to
# -Number of 1-4 family homes

#Derived Assumptions
#
#1. Interest Rate and Tract Population propbably not important.
#
#2. Number of Occupied Units seems to be connected to the Tract perc
#   of regional median income and the perc of minority population, 
#   number of 1-4 family homes and Tract population. Therefore, it
#   probably won't add much itself. Whatever effect it has will likely
#   be captured in those variables.
#
#3. Number of 1-4 family homes can capture some of what population does
#   and is otherwise not very correlated. It can probably be used as a
#   stand in for the effects of population and contribute its own as well.
#   So, a community with fewer apt buildings may have more exp homes.
#
#4. Tract Minority Population Percentage and Tract perc of Regional
#   Median Family Income are highly correlated (-) with each other
#   and Tract perc of Regional Median Family Income looks to capture
#   more of Number of Owner Occupied Units, so may be the more important
#   variable to keep.
#
#5. Best to use Tract perc of Regional Median Family Income and
#   Number of 1-4 family homes as explanatory variables for
#   Propety Value. VIF tests will be needed to check this.

#pairplot
#sns.pairplot(filtered_popval.iloc[:, np.r_[0,6:12]])

##################################################

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

#Top-Level
print('\n---\nTop-Level\n---\n')
for col in filtered_popval.iloc[:, np.r_[0,6:12]].columns:
    print(f'\n{col}\n')
    make_hist_kde(filtered_popval, x=col, bins=bins, hue=None)

#Grouped by Occupancy Type
print('\n---\nGrouped by Occupancy Type\n---\n')
for col in filtered_popval.iloc[:, np.r_[0,6:12]].columns:
    print(f'\n{col}\n')
    make_hist_kde(filtered_popval, x=col, bins=bins, hue='occupancy_type')

# print('\nProperty Value')
# make_hist_kde(filtered_popval, x='property_value', bins=bins)
# print('\nTract Population')
# make_hist_kde(filtered_popval, x='tract_population', bins=bins)
# print('\nPercentage of Tract Population that is in Minority Group')
# make_hist_kde(filtered_popval, x='tract_minority_population_percent', bins=bins)
# print('\nRegional Median Family Income')
# make_hist_kde(filtered_popval, x='ffiec_msa_md_median_family_income', bins=bins)
# print('\nTract Median Family Income as Percentage of Regional Median Family Income')
# make_hist_kde(filtered_popval, x='tract_to_msa_income_percentage', bins=bins)
# print('\nNumber of Owner Occupied Units')
# make_hist_kde(filtered_popval, x='tract_owner_occupied_units', bins=bins)
# print('\nNumber of 1-4 Family Homes')
# make_hist_kde(filtered_popval, x='tract_one_to_four_family_homes', bins=bins)
# print('\nMedian Age of Units')
# make_hist_kde(filtered_popval, x='tract_median_age_of_housing_units', bins=bins)

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
sns.pairplot(filtered_popval.iloc[:, np.r_[6,7,9:12]], plot_kws={'alpha': 0.1})
plt.show()

##################################################

# print('\nFreq plots for cat vars...\n')

#Only occupancy type had some kind of distribution
#was captured in Top-Level stacked histograms by occupancy_type

##################################################

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
#Model with all numerical values chosen
print('\nModel with all numerical variables chosen\n')
model_test=ols(formula='property_value ~ tract_population + tract_minority_population_percent + ffiec_msa_md_median_family_income + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())

#%%
#Suspected important values
print('\nModel For Suspected Important Factors, With VIF less than 10\n')
#derived_dwelling_category + occupancy_type + construction_method + total_units + interest_rate
#+ tract_population + tract_minority_population_percent + ffiec_msa_md_median_family_income + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units'

#Build model
model_test=ols(formula='property_value ~ C(occupancy_type) + tract_minority_population_percent + tract_to_msa_income_percentage + tract_one_to_four_family_homes', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())

#%%
#Model all vars that showed low p-value (final?)
print('\nModel For all Factors that had a low p-value in the All-Factors-Model\n')
#+ C(total_units)
model_test=ols(formula='property_value ~ C(occupancy_type) + tract_minority_population_percent + tract_to_msa_income_percentage + tract_owner_occupied_units + tract_one_to_four_family_homes + tract_median_age_of_housing_units', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())

#%%
#Model No cat vars, just low-enough-VIF numeric cars
print('\nModel of Only Lower VIF Factors\n')
model_test=ols(formula='property_value ~ tract_minority_population_percent + tract_to_msa_income_percentage + tract_one_to_four_family_homes', data=filtered_popval)
model_test_fit=model_test.fit()
print(model_test_fit.summary())

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

######################
## CODE CEMETARY :( ##
######################

######################################
## Graphical Exploration: Inital DF ##
######################################

# print('\nGraphical Exploration: Initial DF...\n')

    #*******************************#
##### INSERT Illiyuna's graphs here #####
    #*******************************#

#A grouped bar chart with the average loan value by loan 
# type. Label the averages for each bar on the chart.

# Used construction method as a proxy from property type 
# Loan Type
# 1 -- Conventional (not insured or guaranteed by FHA VA RHS or FSA)
# 2 -- Federal Housing Administration insured (FHA)
# 3 -- Veterans Affairs guranteed (VA)
# 4 -- USDA Rural Housing Service or Farm Service Agency guaranteed (RHS or FSA)

# sns.barplot(data=init_df, 
#             x='loan_type', 
#             y='loan_amount', 
#             hue='construction_method', ci=None)
# plt.xlabel('Type of Loan',size=14)
# plt.ylabel('Average Loan Amount',size=14)
# plt.title('Average Loan Amount & Type by Property Type') 
# x_var= ['Conventional', 
#         'FHA Insured', 
#         'VA Insured',
#         'RHS or FSA Insured']
# plt.xticks([0,1,2,3], x_var, rotation=20)
# labels= ['Site Built', 'Manufactured Home']         
# plt.legend(labels,loc='upper left')
# plt.show()


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

# %%
