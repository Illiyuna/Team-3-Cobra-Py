#%%

#############
## Imports ##
#############

from operator import index
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

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