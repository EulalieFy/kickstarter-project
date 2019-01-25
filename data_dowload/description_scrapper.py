# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:48:00 2019

@author: damie
"""

import os
import re
import newspaper
from tqdm import tqdm


def get_articles(url):

    ''' Download un article a partir d'un URL grace au module *newspaper* '''
    
    art = newspaper.Article(url)
    art.download()
    art.parse()
    return(art.text)
    
DEFAULT_PATH = './data/Description'

def download_description(row):
    
    if os.path.exists(os.path.join(DEFAULT_PATH,'{}.txt'.format(row['id']))) == True:
        print('Already Downloaded')
        return

    try :
        url = re.findall(r'(?<="project":")[^"]*', row['urls'])[0]
        article = get_articles(url)
        #print(row['id'])
        with open(os.path.join(DEFAULT_PATH, '{}.txt'.format(row['id'])), 'w') as f:
            f.write(article)
            f.close()
    except :
        print('Error while downloading ID:', row['id'], 'url', url)
        
def download_from_dataframe(df):
    for row in tqdm(df.iterrows()):
        download_description(row[1])
    
#print(get_articles('https://www.kickstarter.com/projects/529454320/the-story-of-pweep-from-egg-to-peacock?ref=category_newest'))