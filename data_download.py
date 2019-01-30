

# !pip install googledrivedownloader

import os
import progressbar

from google_drive_downloader import GoogleDriveDownloader as gdd

"""
Download data from a google drive, create a 'data' folder and 
copy the four csv files into it

The file_ids must refer to files with 'public' option in 'share' parameter 
"""

data_and_labels = [ 'data_train.csv', 'data_test.csv',
                    'labels_train.csv', 'labels_test.csv']

dic_ids = {'data_train.csv': '1FBYmIGguFp3fXo6eiMNBMXLHGaLlXydQ',
           'data_test.csv':'1qwnQ-X6oSMaRRsjf-9E5j7eGMaeAxcha',
           'labels_train.csv':'1SNnnBNi_hpyMBXCX742BfYtvqE8ANe1T',
           'labels_test.csv':'1j07uF2EP67bvjNbWCA1EzFAw0vZBNoW9'}


def main(output_dir='data'):

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
        
if __name__ == '__main__':

    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()
        
