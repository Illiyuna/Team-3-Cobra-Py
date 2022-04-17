#%%

#imports
from operator import index
import pandas as pd
from IPython.display import display



#%%
################################
## Test filtering for Dc data ##
################################

#first just see what is in a chunk
#load chunk1
first_chunk_df=pd.read_csv("chunk1.csv")

#%%
#view chunk1 head
display(first_chunk_df.head())

# %%
#count number of observation in chunk1 (should be 50k)
print(f'number of observation in chunk: {len(first_chunk_df)}')

#%%
#test filtering chunk1 only for DC data
cond=first_chunk_df['state_code']=='DC'

#show how many dc observarion in chunk 1 and display dc_subset's head
print(f'number of DC observations in chunk: {len(first_chunk_df[cond])}')
display(first_chunk_df[cond].head().reset_index())

#########################################################################################

#%%

# Ok, that looks good, so let's run through all chunks and make a full DC filtered df

#########################################################################################

# %%
#Get initial dc_df from first chunk

chunk_df=pd.read_csv(f"chunk1.csv")
cond=chunk_df['state_code']=='DC'
dc_df=chunk_df[cond].reset_index()
display(dc_df.head())

#%%
#Loop through rest of chunks, filter for DC, append to Dc df
for i in range(1,512):
    #get dc rows from current chunk
    chunk_df=pd.read_csv(f"chunk{i}.csv")
    cond=chunk_df['state_code']=='DC'
    chunk_df=chunk_df[cond]

    #add dc rows from current chunk to full dc_df
    dc_df=pd.concat([dc_df, chunk_df])
    print(f'Dc observations: {len(dc_df)}')

#%%
#format dc_df

dc_df=dc_df.reset_index()
dc_df=dc_df.drop(['index'], axis=1)
dc_df=dc_df.drop(['level_0'], axis=1)

#Show dc_df
display(dc_df)

# %%
#Write back to csv for use
dc_df.to_csv('dc_df.csv',index=False)

# %%
