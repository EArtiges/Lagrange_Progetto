from datetime import datetime
from dateutil import parser
import json
import time
import gzip
import pandas as pd
import twython
from twython import TwythonRateLimitError
import Twitter_scraping as scraper
import string
import pickle as pkl
import numpy as np
from requests.exceptions import Timeout, ConnectionError
from requests.packages.urllib3.exceptions import ReadTimeoutError

import logging
logging.basicConfig(filename='Twitter.log',level=logging.DEBUG)

APP_KEY = 'ZIvxV0lZJnQI2FOuEqi0zzQqU'
APP_SECRET = '28z7HZuCNX71NMc5TO8ga3woravEt5wFsvm7Z2Q7LERBpoSCno'
twitter = twython.Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()
twitter = twython.Twython(APP_KEY, access_token=ACCESS_TOKEN)

print('Grid')
# Define the grid we want to work with.
Grid=pkl.load(open('../Results/2019-02-05_8_5000_diy_word2vec/pickle/Grid.pkl', 'rb'))
latitudes=set([g[0] for g in Grid])
longitudes=set([g[1] for g in Grid])
Grid=[]
for lat in sorted(latitudes):
    for lon in sorted(longitudes):
        Grid.append((lat,lon))
grid_str=[str(g[0])+','+str(g[1])+',0.35km' for g in Grid]

print('queries')
# Run the queries we want on the grid we choose
statuses_all=[]
user_loc_dict={}
for query in ['']:
#'torino','san salvario','nizza','porta nuova','piazza castello','mole','vanchiglia','vanchiglietta','lingotto','san donato','borgo crimea','bordo po','santa rita','mirafiori','crocetta','cenisia']:
    statuses=[]
    for g in grid_str:
        last_id=None
        try:
            search_parameters = dict(q=query , count = 100, max_id = last_id, geocode=g)
        except (ReadTimeoutError, ConnectionError, ConnectionResetError, TwythonRateLimitError,OSError) as exc:
            logging.exception("message")
            continue
        print("Starting to query tweets")
        statuses+=scraper.query_by_word(search_parameters, twitter)
    statuses_all+=statuses
    print('filtering')
    # For all the tweets we got, keep only the interesting info
    tweets=[]
    for status in statuses:
        rt = 'retweeted_status' in status
        text = status['retweeted_status']['text'] if rt else status['text']
        status['geo'] = status['retweeted_status']['geo'] if rt else status['geo']
        try:
            loc=status['geo']['coordinates'] if status['geo'] else status['query_loc']
            loc=','.join([str(i) for i in [loc]])
        except KeyError:
            loc=status['query_loc']
        tweets.append((np.uint64(status['id_str']),
                       text,
                       rt,
                       status['created_at'],
                       status['lang'],loc,
                       np.uint64(status['user']['id_str'])))

    print('making/pickling dataframe')
    # Store everything in a dataset and pickle it
    df = pd.DataFrame(tweets,
                  columns=['status_id', 'text', 'rt', 'created_at', 'lang', 'location','user_id',])
    df = df.set_index('status_id')
    df.created_at = pd.to_datetime(df.created_at)
    df.to_pickle('DF/dataframe_query_{}.pkl'.format(query))
    
    print('list of users')
    #Make a list of users and build their spatial profiles
    user_loc=[e for e in zip(df.user_id.tolist(), df.location.tolist())]
    for e in user_loc:
        if not e[0] in user_loc_dict:
            user_loc_dict[e[0]]=[e[1]]
        else: 
            user_loc_dict[e[0]].append(e[1])

for e in user_loc_dict:
    user_loc_dict[e]=scraper.spatial_profile(user_loc_dict[e])
        
print('pickling dictionary')
# Save the dicionary of users spatial profiles
pkl.dump(user_loc_dict, open('user_loc_dict.pkl', 'wb'))


print("Query executed. Saving files in a JSON")
with gzip.open('torino_2019.json.gz', 'wt') as gf:
    for status in statuses_all:
        gf.write(json.dumps(status) + "\n")
        
print("Starting to query users.")
# Scrape users timelines
ID = list(user_loc_dict.keys())
date_start = parser.parse('2000-02-01 00:00:00+00:00')
print(len(ID), 'users to scrape')

for user_ID in ID:
    print("scraping user {}".format(user_ID))
    while(True):
        try:
            df=scraper.get_user_timeline(user_ID,date_start,twitter)
            df.to_pickle('DF/USERS/dataframe_user_{}.pkl'.format(user_ID))
        except (ReadTimeoutError, ConnectionResetError, ConnectionError,TwythonRateLimitError,OSError) as exc:
            logging.exception("message")
            continue
        break
       

