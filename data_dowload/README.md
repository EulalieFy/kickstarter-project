# Data Downloader

We are using data scrapped from the website 'www.kickstarter.com', made available by  webrobots (https://webrobots.io/kickstarter-datasets/) and 
hosted on a AWS S3 instance. 

![ScreenShot](website.png)

Hence, we're using *boto3* packages on Python so as to download it manually and store it directly on a local machine without having to store it on a cloud.
Once the data is downloaded, we unzip the dataframe and concatenante all the cvs at our disposal into one.

One can use the following command line to run the downloaded 

```python
from kickstarter_datadoawloaded import start_downloading
strat_downloading() 
```
