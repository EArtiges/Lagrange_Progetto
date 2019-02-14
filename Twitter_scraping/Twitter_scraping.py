from datetime import datetime
from dateutil import parser
import json
import time
import gzip
import pandas as pd
import twython
from twython import TwythonRateLimitError
import numpy as np
import pickle as pkl

def query_by_word(search_parameters,twitter):
    searching = True
    last_id=None
    statuses=[]
    while (True):
        # check for rate limit
        time.sleep(.2)
        #search_parameters = dict(q="#sansalvario" , count = 100, max_id = last_id, bounding_box='7.672325,45.050224,7.684212,45.054893')
        print ("Searching with parameters: ", search_parameters)
        # search for tweets
        try:
            result = twitter.search(**search_parameters)
        except TwythonRateLimitError:
            time.sleep(901)
            continue
        # add the statuses to a list
#        print(len(result['statuses']), "tweets collected")
        if len(result['statuses'])==0:
            break
        else:
            for status in result['statuses']:
                status['query_loc']=search_parameters['geocode']
                status['user']['query_loc']=search_parameters['geocode']
                statuses.append(status)
                last_result = status
            # update max_id
        last_id = int(last_result['id_str']) - 1
        search_parameters['max_id']=last_id
    print(len(statuses), "tweets collected.")
    return statuses


def get_user_timeline(user_ID,date_start,twitter):
    #print('searching for user {}'.format(user_ID))
    searching = True
    last_id=None
    statuses=[]
    while (searching):
        time.sleep(.2)
        try:
            result = twitter.get_user_timeline(id = user_ID, count = 200, include_rts = True, max_id=last_id)
        except TwythonRateLimitError:
            time.sleep(901)
            continue
        if len(result) == 0:
            break
        # add the statuses to a list
        for status in result:
            created_at = parser.parse(status['created_at'])
            if created_at < date_start:
                # if created_at is less than date_start, stop searching
                searching = False
                break
            statuses.append(status)
            last_result = status
        # update max_id
        last_id = int(last_result['id_str']) - 1

    tweets = []

    for status in statuses:
        rt = 'retweeted_status' in status
        text = status['retweeted_status']['text'] if rt else status['text']
        try:
            loc=status['retweeted_status']['location']+'*rt' if not status['location'] else status['location'][0]+'*tweet'
        except KeyError:
            loc=status['user']['location']+'*user'
        tweets.append((np.uint64(status['id_str']),
                       text,
                       rt,
                       status['created_at'],
                       status['lang'],loc,
                       np.uint64(status['user']['id_str'])))
    df = pd.DataFrame(tweets,
                  columns=['status_id', 'text', 'rt', 'created_at', 'lang', 'location','user_id',])
    df = df.set_index('status_id')
    df.created_at = pd.to_datetime(df.created_at)
    with open('USERS/timeline_user_{}.pkl'.format(user_ID), 'wb') as f:
        pkl.dump(statuses,f)
    return df
        
def spatial_profile(locs):
    try:
        return [(l,len([i for i in locs if i==l])) for l in set(locs)]
    except TypeError:
        loc2=[]
        for l in locs:
            if l['coordinates']:
                l=l['coordinates']
            loc2.append(l)
        locs=loc2
        return [(l,len([i for i in locs if i==l])) for l in set(locs)]

