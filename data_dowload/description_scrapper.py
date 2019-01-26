# -*- coding: utf-8 -*-
"""
Script that scraps text descriptions from the project's home page on 
Kickstarter ans save it in a txt file 

"""

import os
import re
import newspaper
from tqdm import tqdm

#Make sure, the data folder already exists
assert os.path.exists('./data')

# If the Image folder doesn't exists, we create one
if os.path.exists('./data/Description') == False : 
    os.mkdir('./data/Description')


def get_articles(url):

    ''' Download un article a partir d'un URL grace au module *newspaper* '''
    
    art = newspaper.Article(url)
    art.download()
    art.parse()
    return(art.text)
    
PATH_TO_TEXT = './data/Description'

def download_description(row):
    if os.path.exists(os.path.join(PATH_TO_TEXT,'{}.txt'.format(row['id']))) == True:
        print('Already Downloaded')
        return

    try :
        url = re.findall(r'(?<="project":")[^"]*', row['urls'])[0]
        article = get_articles(url)
        #print(row['id'])
        with open(os.path.join(PATH_TO_TEXT, '{}.txt'.format(row['id'])), 'w') as f:
            f.write(article)
            f.close()
    except :
        print('Error while downloading ID:', row['id'], 'url', url)
        
def download_from_dataframe(df):
    
    ''' Iterate on the rows of the dataframe to get most descriptions '''
    
    for row in tqdm(df.iterrows()):
        download_description(row[1])
    
#print(get_articles('https://www.kickstarter.com/projects/529454320/the-story-of-pweep-from-egg-to-peacock?ref=category_newest'))