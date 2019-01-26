# -*- coding: utf-8 -*-
"""
Script to get images from url and save it into a ./data/Image folder

"""

import re
import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import IPython.display as ipd

#Make sure, the data folder already exists
assert os.path.exists('./data')

# If the Image folder doesn't exists, we create one
if os.path.exists('./data/Image') == False : 
    os.mkdir('./data/Image')

def get_picture(row,size ='big'):
    if size == 'big':
        print(row['profile'])
        return re.findall(r'(?<="default":")[^"]*', row['profile'])[0]
    if size == 'small':
        return re.findall(r'(?<="baseball_card":")[^"]*', row['profile'])[0]

#get_picture(df1.loc[1,:], 'small')

PATH_TO_IMAGE = './data/Image'

def download_from_pandas(df):
    
    for row in tqdm(df.iterrows()):
    
        if os.path.exists(os.path.join(PATH_TO_IMAGE,'{}.png'.format(row[1]['id']))) == True:
            print('Already Downloaded')
            continue

        try :
            url = get_picture(row[1],'small')
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save(os.path.join(PATH_TO_IMAGE,"{}.png".format(row[1]['id'])))
        except : 
            print('Some error occurred with ID :', row[1]['id'])
        #ipd.display(img)
