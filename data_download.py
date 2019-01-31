###############################################################################
############################### Data Downloader ###############################
###############################################################################

''' Running this file will automatically start the download of the data.
    You can either just run the following cmd line if you just need the dataframes
        - python data_downloader
    Or you can as well add flags --images True or --descriptions True (or both) :
        - python data_downloader --images True --descriptions True
    As a remark, the dataframes and the 30 000 descriptions are relatively quick
    to be downloaded wherease images can take a while (remove the flag --images True 
    in this case) '''
    
import os
import tqdm
import zipfile
import argparse

from google_drive_downloader import GoogleDriveDownloader as gdd

"""
Download data from a google drive, create a 'data' folder and 
copy the four csv files into it

The file_ids must refer to files with 'public' option in 'share' parameter 
"""

data_and_labels = [ 'data_train.csv', 'data_test.csv',
                    'labels_train.csv', 'labels_test.csv',
                    'full_data.csv']

dic_ids = {'data_train.csv': '1FBYmIGguFp3fXo6eiMNBMXLHGaLlXydQ',
           'data_test.csv': '1qwnQ-X6oSMaRRsjf-9E5j7eGMaeAxcha',
           'labels_train.csv': '1SNnnBNi_hpyMBXCX742BfYtvqE8ANe1T',
           'labels_test.csv': '1j07uF2EP67bvjNbWCA1EzFAw0vZBNoW9',
           'full_data.csv': '1x32C5QfLnamLf-Nn3zRBfUY50c-3D_dg'}


def main(output_dir='data', images = False, descriptions = False):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print('Data Folder Created')
    
    print("Downloading from Google Drive ...")
    for csv in data_and_labels:
        
        destination = './{}/{}'.format(output_dir, csv)
        print(destination)
        
        if os.path.exists(destination):
            print("... File  {} already downloaded".format(destination))
            continue
    
        gdd.download_file_from_google_drive(file_id=dic_ids[csv],
                                            dest_path=destination,
                                            unzip=False)
        
        
        print("... File saved as {}".format(destination))
        
    if descriptions == True:
        
        if os.path.exists('./{}/Descriptions'.format(output_dir)):
            print("... Descriptions already downloaded")
            
        else :
            
            if not os.path.exists('./{}/Descriptions'.format(output_dir)):
                os.mkdir('./{}/Descriptions'.format(output_dir))
           
            ids = '1Y3Qh4BFKnAOptSS3QblRNFeKDsOfrY0s'
            destination = './{}/Descriptions/{}'.format(output_dir, 'descriptions.zip')
            
            gdd.download_file_from_google_drive(file_id=ids,
                                                dest_path=destination)
            
            zip_ref = zipfile.ZipFile(destination, 'r')
            zip_ref.extractall('./{}/Descriptions/'.format(output_dir))
            zip_ref.close()
            
            os.remove(destination)
            
            print('... 30 000 Descriptions downloaded')
            
    if images == True :
        
        if os.path.exists('./{}/Images'.format(output_dir)):
            print("... Images already downloaded")

        else :
            
            if not os.path.exists('./{}/Images'.format(output_dir)):
                os.mkdir('./{}/Images'.format(output_dir))
            
            ids_list = ['1Gm830WcZLyP86xYvz_QOiWi7mBCUzgTn', '1wIy-jxoCDvseJbgRQPQvuqGxJy9sy155',
                        '1wgG9faO-PbkwmjZPj3Kj6v5I8PYRgNwH', '1q3t9xLJZ5iEBBlJK644-Qy1_O15Rkmr_',
                        '1rNOWsJ3gyYBGPzBF_QFRacUyKtu5AUAd', '1jDzhane_xNe-G-cykP7JY5BfTlGDYxM0']
                        
            for ids in tqdm.tqdm(ids_list):
                
                print('\n5 000 Images downloading ...')
                
                destination = './{}/Images/{}'.format(output_dir, 'Images.zip')
                 
                gdd.download_file_from_google_drive(file_id=ids,
                                                     dest_path=destination)
            
                zip_ref = zipfile.ZipFile(destination, 'r')
                zip_ref.extractall('./{}/Images/'.format(output_dir))
                zip_ref.close()
                
                os.remove(destination)
                
            print('... 30 0000 Images downloaded')
        
    
        
if __name__ == '__main__':
    
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--images", help="Whether or not you want to download 30000 images",  type=bool, default = False)
        parser.add_argument("-d", "--descriptions", help="Whether or not you want to download 30000 descriptions",  type=bool, default = False)
        args = parser.parse_args()
        main(images = args.images, descriptions = args.descriptions)