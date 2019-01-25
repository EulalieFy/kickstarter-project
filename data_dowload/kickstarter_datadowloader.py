# -*- coding: utf-8 -*-
"""
Downloader for KickStarter Data
"""

#Packages
import os
import boto3
import zipfile
import progressbar
from botocore.handlers import disable_signing

''' Change path here maybe '''
print(os.getcwd())

# Données à télécharger
link = 'https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2018-12-13T03_20_05_701Z.zip'

# From AWS S3 instances, using BOTO3 module
resource = boto3.resource('s3')
resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)

BUCKET_NAME = 'weruns' # replace with your bucket name
KEY = 'forfun/Kickstarter/Kickstarter_2018-12-13T03_20_05_701Z.zip' # replace with your object key
bucket = resource.Bucket(BUCKET_NAME)

with open('Kickstarter_data.zip', 'wb') as data:
    file_name = os.path.join(os.getcwd(),'Kickstarter_data.zip')
    statinfo = os.stat(file_name)
    up_progress = progressbar.progressbar.ProgressBar(maxval=204864714)#statinfo.st_size)
    up_progress.start()

    def upload_progress(chunk):
        up_progress.update(up_progress.currval + chunk)
        
    bucket.download_fileobj(KEY, data,  Callback=upload_progress)
    
# Get size of final zip file
#statinfo = os.stat(file_name)
#statinfo.st_size

#Unzip
zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall('./data/Kickstarter')
zip_ref.close()

print('Process Done')