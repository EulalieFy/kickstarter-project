

# !pip install googledrivedownloader

import os
import progressbar

from google_drive_downloader import GoogleDriveDownloader as gdd

"""
Download data from a google drive, create a 'data' folder and 
copy the four csv files into it

The file_ids must refer to files with 'public' option in 'share' parameter 
"""

####### Faut le faire avec data_test et data_train maintenant

data_and_labels = [ 'data_train.csv', 'data_test.csv',
                   'labels_train.csv', 'labels_test.csv']

dic_ids = {'data_train.csv': '1V8q7MhLBUD7C2Ny24SAa343U5BIXIFE_',
          'data_test.csv':'1DSnvNcMXjw9jN9m0H63zhSyuzq8UmwiM',
          'labels_train.csv':'11baOLEPh_1hmIxjlNnBt0t1jsVDj4l_2',
          'labels_test.csv':'1jbDisLW0w2YpzHwvvc3V7T4e_gbKMzYV'}


def main(output_dir='data'):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print('Data Folder Created')


    #file_ids = ['1kZVi3f6pHq3wyt4FpR_8-c4wwYgykL6o']
    
    #destinations = ['./{}/{}.csv'.format(output_dir, key)]
    #print(destinations[0], os.path.exists(destinations[0]))
    
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
        
