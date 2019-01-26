

from google_drive_downloader import GoogleDriveDownloader as gdd


"""
Download data from a google drive, create a 'data' folder and 
copy the four csv files into it

The file_ids must refer to files with 'public' option in 'share' parameter 
"""


file_ids = ['1ef8FfQVnL7tgZcUE4deacqPZZpGRDFs2']

destinations = ['./data/test.csv']

for file_id, destination in zip(file_ids, destinations):

    gdd.download_file_from_google_drive(file_id=file_ids,
                                        dest_path=destination,
                                        unzip=False)