# Team 3 -- Cobra Py
# Overcharged

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

#%% Used to Unzip file & Set Working Directory 

# from zipfile import ZipFile
# with ZipFile('2018_public_lar_csv.zip', 'r') as zipObject:
#    listOfFileNames = zipObject.namelist()
#    for fileName in listOfFileNames:
#        if fileName.endswith('.csv'):
#            # Extract a single file from zip
#            zipObject.extract(fileName, 'temp_py')
#            print('All the python files are extracted')

# os.getcwd()
# os.chdir('/Users/illiyunaislam/OneDrive/Grad School/Courses/Data Mining/Team-3-Cobra-Py')
# os.getcwd()

#%%
# Import Data 
df = pd.read_csv('dc_df.csv')

#%%
df.describe()
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
plt.title('Action Taken')
plt.xlabel(' Taken')
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

###################################################################################################################

#                                        RACE & GENDER 

#%%
# Overcharged subset
print('\nDisplay Summary statistics for Overcharged DF...\n')

overcharged_df=overcharged_df.dropna()
overcharged_df.isnull().sum() / overcharged_df.shape[0] * 100
# overcharged_df[loan_cond].describe()
# overcharged_df[intr_cond].describe()
# overcharged_df[intr_cond2].describe()
# overcharged_df[combined_cond].describe()

# Dropping "Sex Not Available" From Derived Sex
overcharged_df = overcharged_df[overcharged_df.derived_sex != 'Sex Not Available']
overcharged_df = overcharged_df[overcharged_df.derived_race!= 'Race Not Available']

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


#%%

###########################################
## Graphical Exploration: Overcharged DF ##
###########################################

print('\nGraphical Exploration: Overcharged DF...\n')

#Top-level graphs
print('\nOvercharged DF Top-level Graphs')

#barplot of race
print('\nBar plot of Race')
labs=['White', 'Black or African American',
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
# Correlation Matrix for Overcharged 

corrMatrix = overcharged_df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()






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

# discount points hist
print('\nDiscount Points Distribution')
sns.histplot(data=overcharged_df, x='discount_points', bins=50)
plt.show()

# lender credits hist
print('\nLender Credits Distribution')
sns.histplot(data=overcharged_df, x='lender_credits', bins=50)
plt.show()

#regplot interest rate against total loan cost
print('\nScatterplot with LR line, Interest Rate Against Total Loan Cost')
g=sns.regplot(data=overcharged_df[combined_cond], x='total_loan_costs',
                y='interest_rate', scatter_kws={'alpha':0.03}, color="gray")
g.lines[0].set_color("pink")
plt.show()

#regplot discount points vs lender credits 
print('\nScatterplot with LR line, Discount Points Against Lender Credits')
g=sns.regplot(data=overcharged_df, x='lender_credits',
                y='discount_points', scatter_kws={'alpha':0.03}, color="gray")
g.lines[0].set_color("pink")
plt.show()

#regplot discount points vs interest rate 
print('\nScatterplot with LR line, Interest Rate Against Discount Points')
g=sns.regplot(data=overcharged_df, x='discount_points',
                y='interest_rate', scatter_kws={'alpha':0.03}, color="gray")
g.lines[0].set_color("pink")
plt.show()

#regplot total loan costs vs lender credits 
print('\nScatterplot with LR line, Total Loan Costs Against Lender Credits')
g=sns.regplot(data=overcharged_df, x='lender_credits',
                y='total_loan_costs', scatter_kws={'alpha':0.03}, color="gray")
g.lines[0].set_color("pink")
plt.show()

#regplot interest rate vs lender credits 
print('\nScatterplot with LR line, Interest Rates Against Lender Credits')
g=sns.regplot(data=overcharged_df, x='lender_credits',
                y='interest_rate', scatter_kws={'alpha':0.03}, color="gray")
g.lines[0].set_color("pink")
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



### What is the probability of getting a high cost loan based on your sex & race??
### Are men getting more or bigger loans than women? 
### Blacks get more credits but still get higher IR? Hypocrisy of financial inst.


# Anovas for everything 
# Tukey post hoc 



# Model: 

#1. dep: IR, loan amount, 