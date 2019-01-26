# -*- coding: utf-8 -*-
"""
KickStarter and Features Selection/Processing
"""

''' First of all, after downloading the data and unzip it,
 we need to merge all the dataframes at our disposal into a big one '''

PATH_TO_DATA ='.'

import os 
import re
import ast
import numpy as np
import pandas as pd
from datetime import datetime

print(os.listdir(PATH_TO_DATA))
df_all = pd.read_csv(os.path.join(PATH_TO_DATA,'Kickstarter.csv'))

for i in range(1,55) :
    ind ="{:03d}".format(i)
    df_all = pd.concat([df_all, pd.read_csv(os.path.join(PATH_TO_DATA,'Kickstarter{}.csv'.format(ind)))])

# Set a new index
df_all.set_index([[i for i in range(df_all.shape[0])]], inplace =True)
df_all.to_csv(os.path.join(PATH_TO_DATA,'Kickstarter_all.csv'))

''' Then, as the information in the CSV is a bit in disorder, we need to
select the data we consider useful for the task '''

df = pd.read_csv('Kickstarter_all.csv', index_col=0)
print(df.shape)
df.head()

# We rename some variables to make it clearer and without confusion
dict_rename = {'backers_count': 'backers',
              'blurb' : 'short_description',
              'name' : 'name',
              'staff_pick' : 'project_we_love',
              'state' : 'state',
              'country':'country',
              'category':'category_dict', 
              'location':'location'}

df.rename(columns = dict_rename, inplace=True)

# Then we extract the information that is hidden within str variable - most
# of them are dictionnary defined as a big string, not very convenient.
# Hopefully using , be it ast.literal_eval function or REGEX matching expression
# the job is made easier.

# Category
df['main_category']=df.apply(lambda row :   ast.literal_eval(row['category_dict'])['slug'].split('/')[0]  , axis=1 )
df['category']=df.apply(lambda row :  ast.literal_eval(row['category_dict'])['slug'].split('/')[-1] , axis=1 )

# Important DateTime related to the project
df['created_at'] =  df['created_at'].apply(lambda row : datetime.fromtimestamp(row))
df['launched_at'] =  df['launched_at'].apply(lambda row : datetime.fromtimestamp(row))
df['state_changed_at'] =  df['state_changed_at'].apply(lambda row : datetime.fromtimestamp(row))
df['deadline'] =  df['deadline'].apply(lambda row : datetime.fromtimestamp(row)) 

# Information about the creator of the project
df['creator_name']=df.apply(lambda row :  ast.literal_eval(row['category_dict'])['name'] , axis=1 )
df['creator_id']=df.apply(lambda row :  ast.literal_eval(row['category_dict'])['id'] , axis=1 )

def get_match(row, pattern):
    
    ''' Return nan value if the cell is empty otherwise 
    compute a regex matching to get some pattern inside a string '''
    
    if pd.isnull(row):
        return row
    else :
        l = re.findall(r'(?<="{}":")[^"]*'.format(pattern), row)
        if len(l) != 0 :
            return l[0]
        else :
            return np.nan
        

# Location information
df['city']= df['location'].apply(lambda row :get_match(row, 'name'))
df['state_location']= df['location'].apply(lambda row : get_match(row, 'state'))

# % of the goal achieved
df['achieved (%)'] = df.apply(lambda row : 100*row['pledged']/row['goal'], axis =1)

# Additionnal variable that we scapped with another script, but not all rows
# have corresponding images or descriptions (if errors occurred for instance)
df['image_available'] = df.apply(lambda row : os.path.exists('./data/Image/{}.png'.format(row['id'])), axis =1)
df['description_available'] = df.apply(lambda row : os.path.exists('./data/Description/{}.png'.format(row['id'])), axis =1)

# Columns we wish to keep for the task
columns = ['id','name', 'short_description', 
           'country', 'city', 'state_location',
           'main_category', 'category', 
           'created_at', 'launched_at', 'state_changed_at', 'deadline', 
           'currency', 'country', 
           'creator_name', 'creator_id',
           'project_we_love', 'image_available', 'description_available',
           'goal', 'pledged','achieved (%)', 'usd_pledged', 'static_usd_rate', 'usd_type', 'state']

df = df[columns]

#df.to_csv(os.path.join(PATH_TO_DATA,'Kickstarter_light.csv'))