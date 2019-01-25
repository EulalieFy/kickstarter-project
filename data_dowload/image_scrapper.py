# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:52:40 2019

@author: damie
"""

import re
import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import IPython.display as ipd

def get_picture(row,size ='big'):
    if size == 'big':
        print(row['profile'])
        return re.findall(r'(?<="default":")[^"]*', row['profile'])[0]
    if size == 'small':
        return re.findall(r'(?<="baseball_card":")[^"]*', row['profile'])[0]

#get_picture(df1.loc[1,:], 'small')

DEFAULT_PATH = './data/Image'

def download_from_pandas(df):
    for row in tqdm(df.iterrows()):
        url = get_picture(row[1],'small')
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(DEFAULT_PATH,"{}.png".format(row[1]['id'])))
        #ipd.display(img)
