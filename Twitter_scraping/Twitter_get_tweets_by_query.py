from datetime import datetime
from dateutil import parser
import json
import time
import gzip
import pandas as pd
import twython
from twython import TwythonRateLimitError
import Twitter_scraping as scraper

APP_KEY = 'ZIvxV0lZJnQI2FOuEqi0zzQqU'
APP_SECRET = '28z7HZuCNX71NMc5TO8ga3woravEt5wFsvm7Z2Q7LERBpoSCno'
twitter = twython.Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()
twitter = twython.Twython(APP_KEY, access_token=ACCESS_TOKEN)

last_id=None
search_parameters = dict(q="#sansalvario" , count = 100, max_id = last_id, bounding_box='7.582325,45.010224,7.764212,45.114893')
statuses=scraper.query_by_word(search_parameters, twitter)


with gzip.open('torino_2019.json.gz', 'wt') as gf:
    for status in statuses:
        gf.write(json.dumps(status) + "\n")


ID = set([s['user']['id'] for s in statuses])
date_start = parser.parse('2000-02-01 00:00:00+00:00')
print(len(ID), 'users to scrape')

for user_ID in ID:
    df=scraper.get_user_timeline(user_ID,date_start,twitter)
    df.to_pickle('USERS/dataframe_user_{}.pkl'.format(user_ID))