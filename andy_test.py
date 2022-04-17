#imports
import pandas as pd

#attempt to use chunkize to load large dataset
#source: https://towardsdatascience.com/loading-large-datasets-in-pandas-11bdddd36f7b

#this should create n number if csv files, which are individually able to be loaded
chunk_size=50000
batch_no=1
for chunk in pd.read_csv('data/2020_lar.csv',chunksize=chunk_size):
    chunk.to_csv('chunk'+str(batch_no)+'.csv',index=False)
    batch_no+=1
