import shelve    
#import gdbm
#mod = __import__('gdbm')
import dbm.dumb
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import datasets, decomposition
import numpy as np
import gc
from datetime import datetime
import datetime as dt
import dateparser
from ipykernel import kernelapp as app
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from operator import itemgetter
import folium
from folium.plugins import HeatMap
import math
import re
import string
import emoji
import pickle
import scipy.io
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem
import sktensor
import geopy.distance
from itertools import groupby
import ncp_py3
import os
from datetime import timedelta
from dateutil.relativedelta import *
from nltk.corpus import stopwords
import nltk.data #
nltk.download('punkt')
import scipy.io
from scipy.io import savemat
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import string



words_monuments=['parcodelvalentino','valentino','parcodora','dora','venaria',
                 'venariareale','reggiadivenaria','reggia','reale',
                 'mole','moleantonelliana','antonelliana','museodelcinema',
                 'piazzavittorio','monteideicappucini','monteideicappucino','museoegizo','museoegizio',
                 'palaalpitour','palazzomadama','palazzoreale',
                'piazzacastello','piazzasancarlo','piazzavittorio','superga']
words_added=['friends', 'italia','food','igers','igerstorino','cosa','bestoftheday',
             'cosi','sole','instagood','picoftheday','followme','instalike',
             'likelike','focus_on','instagram','photooftheday','instadaily',
             'torino','love','italy','turin','kappafuturfestival','kff','torinocomics',
            'lingotto','followfollow','foodporn','instafood','foodgasm','stadium',
             'juventusstadium','tagsforlikes','instamood','follow','madre','granmadre']
words_filtered=list(set(['juventus', 'finoallafine', 'juventusstadium', 'juve',
'stadium', 'forzajuve', 'football', 'serie', 'bianconero', 'championsleague','instaitaly',
'solo','cosa','cosi','oggi','allianzstadium','antonelliana','followback','followers',
'followforfollow','foodblogger','foodlover','foodpic','foodpics','foodstagram',
'igdaily','igersitalia','igersitaly','igersoftheday','igerspiemonte','igersturin',
'igeuropa','igitalia','igitaly','igpiemonte','igtorino','igtravel','igturin','igworldclub',
'insta','instaart','instacool','instaday','instafashion','instafollow','instag',
'instagramers','instagramhub','instagrammers','instaitalia','instalife','instalove',
'instamoment','instamusic','instaphoto','instapic','instatorino','instatravel',
'instatraveling','iphoneonly','iphone','iphones','iphoneshoot','iphonesia','iphonestyle','iphoneplus', 'iphonephoto', 'italia','italiainunoscatto','italian','italianfood','italiangirl','italianplaces','italianstyle','like','likeback','likefollow',
'likeforfollow','likeforlik','likes','likesforlikes','likeslikes','palestro',
'sansalvario','salonedelgusto','tag','tags','tagsforlikesapp',
'travel','travelblogger','travelgram','traveling','travelingram','travelling',
'travelphotography','vittorio','webstagram','\u0438\u0442\u0430\u043b\u0438\u044f',
'\u043d\u0430','\u043d\u0435','\u0442\u0443\u0440\u0438\u043d','\u0447\u0442\u043e','beautiful','smile','torino\xc3\xa8lamiacitt']))

french_stopwords = set(stopwords.words('french'))
english_stopwords = set(stopwords.words('english'))
italian_stopwords = set(stopwords.words('italian'))
spanish_stopwords = set(stopwords.words('spanish'))
italian_stemmer = nltk.stem.SnowballStemmer('italian')
all_stopwords=french_stopwords.union(english_stopwords)
all_stopwords=all_stopwords.union(italian_stopwords)
all_stopwords=all_stopwords.union(spanish_stopwords)
all_stopwords=all_stopwords.union(words_added)
all_stopwords=all_stopwords.union(words_filtered)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
                
sentences = MySentences('posts_per_month/') # a memory-friendly iterator

                

def N_posts(df_new, debug=False):
    """Takes a DataFrame and returns the number of posts per months.
    
    The dataset needs to be a pandas DataFrame object with the dates list labbelled date.
    Returns a list of posts and a list of corresponding dates.
    """
    list_dates=list(df_new.date)
    list_dates=[str(l)[0:7] for l in list_dates]
    histo = [[list_dates[0],0]]
    keys=[]
    counter=0
    
    for l in list_dates:
        if l==histo[-1][0]:
            counter+=1
            histo[-1]=[l,counter]
        else:
            counter=1
            histo.append([l,1])
    histo=sorted(histo, key=itemgetter(0))
    if debug:
        print((histo[0:50]))
        
    Histo=[]
    
    for h in histo:
        if h[0] in keys:
            Histo[-1]+=h[1]
        else:
            keys.append(h[0])
            Histo.append(h[1])
    if debug :
        print((Histo[0:20]))
    return keys, Histo

def time_coord(df_location_all, df_new):
    """From two DataFrames, matches the locations from the first with the dates from the second.
    
    The first DataFrame needs a column labelled id for location_id, the second needs a column
    labelled date for the date.
    """
    IDCOORD= dict(list(zip( df_location_all.id,df_location_all.coord)))
    TID = dict(list(zip(df_new.location_id,df_new.date)))
    
    for k in list(IDCOORD.keys()):
        IDCOORD[k] = [TID[k][0:7],IDCOORD[k]]
    TbyCoord=[]
    for k in list(IDCOORD.keys()):
        TbyCoord.append(IDCOORD[k])
    TbyCoord=sorted(TbyCoord, key=itemgetter(0)) 
    return TbyCoord

def clean(df):
    """Removes hyperlinks, quotes, citations, tickers, numbers and punctuation
    
    Documentation about .loc and accessing vs copying here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    """
    temp_df = df.copy()
    #REMOVING DIRTY THINGS
    print('Removing Dirty Stuff...')
    for i,tweet in enumerate(temp_df.text):
        #Remove hyperlinks
        temp = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', str(tweet))
        #Remove quotes
        temp = re.sub(r'&amp;quot;|&amp;amp', '', temp)
        #Remove citations
        temp = re.sub(r'@[a-zA-Z0-9]*', '', temp)
        #Remove tickers
        temp = re.sub(r'\$[a-zA-Z0-9]*', '', temp)
        #Remove numbers
        temp = re.sub(r'[0-9]*','',temp)
        temp_df.text[i] = temp
    #REMOVE PUNCTUATION
    print('Remove Punctuation')
    temp_df.text = [[x for x in tweet if x not in string.punctuation] for tweet in temp_df.text]
    #PORTER STEMMER
    print('Apply Porter Stemmer')
   # porter_stemmer = nltk.stem.PorterStemmer()
   # for i,tweet in enumerate(temp_df.text):
    #  words = nltk.word_tokenize(tweet)
     # for j,word in enumerate(words):
      #  words[j] = porter_stemmer.stem(word)
      #temp_df.text[i] = ' '.join(words)
    return temp_df

def read_CRS(tfidf,n_features,timestamp):
    """Reads a CRS-formatted sparse tensor, and returns a TF_IDF-term/topic timestamped dataframe.
    
    The sparse tensor has to be a sktensor.sptensor object. 
    Number of columns is given by n_features; the number of rows is given by the 
    tfidf data itself. 
    """
    #Column indices
    ListIndices=list(tfidf.indices)
    Nindices=len(ListIndices)
    #How many non zero elements in the i-1 th row
    Ind=tfidf.indptr
    #Actual values
    Dat=list(tfidf.data)
    print(Dat)
    #Create the dataframe to store out data
    data_CRS=pd.DataFrame(np.zeros((Nindices,n_features+1)),columns=[str(i) for i in range(0,1+n_features)])
    for i in range(0,len(tfidf.indptr)-1):
        a=Ind[i]
        b=Ind[i+1]
        B=b-a
        while B !=0 :
            a=str(ListIndices.pop(0))
            b=str(Dat.pop(0))
            data_CRS[a][i]=b
            B-=1
    data_CRS=data_CRS.rename(columns={str(n_features):'date'})
    data_CRS['date'][:]=timestamp
    return data_CRS

def read_CRS_totensor(tfidf,n_features,timestamp):
    """
    Reads a CRS-formatted sparse tensor, and returns an array of non-zero elements coordinates.
    
    The sparse tensor has to be a sktensor.sptensor object. 
    Number of columns is given by n_features; the number of rows is given by the 
    tfidf data itself. 
    startmonth is an int corresponding to the 12-months index of the earliest month in the dataset.
    """
    #Column indices
    ListIndices=list(tfidf.indices)
    Nindices=len(ListIndices)
    #How many non zero elements in the i-1 th row
    Ind=tfidf.indptr
    #print 'ind:', Ind, 'len:', len(Ind)
    #Actual values
    Dat=list(tfidf.data)
    #Create the array to store out data
    data_CRS=[]
    coord_CRS=[]
    #for each row i:
    for i in range(0,len(Ind)-1):
        a=Ind[i]
        b=Ind[i+1]
        B=b-a
        #We have B non zero elements on row i
        while B !=0 :
            #column index
            a=ListIndices.pop(0)
            #value of the element
            b=Dat.pop(0)
            #store the coordinates: row, column and depth value
            coord_CRS.append((i,a,timestamp))
            #and the corresponding value
            data_CRS.append(b)
            B-=1
    return coord_CRS, data_CRS

def abs_to_yearmonth(month_abs,yearmin):
    year_month=str(yearmin+(month_abs-month_abs%12)/12)+' '+str(month_abs%12+1)
    return pd.Timestamp(year_month)

def N_cells_info(df,step_m):
    lat1=min(df.lat)
    lat2=max(df.lat)
    lon1=min(df.lon)
    lon2=max(df.lon)
    # Generate the grid
    step_lon=step_m/(40000*math.cos((lat1+lat2)*math.pi/360)/360)
    step_lat=step_m/(40000./360.)
    # Define the repartition functions...
    to_bin_lon = lambda x: np.floor(x / step_lon) * step_lon
    to_bin_lat = lambda x: np.floor(x / step_lat) * step_lat
    #... and distribute the points to the grid
    df["latbin"] = df.lat.map(to_bin_lat)
    df["lonbin"] = df.lon.map(to_bin_lon)
    #Form the bins
    groups = df.groupby(["latbin", "lonbin"])
    df['coordbin'] = list(zip(df.latbin,df.lonbin))
    #compute the number of rows and columns
    number_lats=(max(df['latbin'])-min(df['latbin']))/step_lat+1
    number_lon=(max(df['lonbin'])-min(df['lonbin']))/step_lon+1
    #So far so good 
    df_index=df[['date','text','coordbin']]
    df_index['date'] = pd.to_datetime(df_index['date'], errors='coerce')
    df_index['year_month'] = list(zip(df_index['date'].dt.year, df_index['date'].dt.month)) 
    #add a month column
    yearmin=min([y[0] for y in df_index.year_month.tolist()])
    df_index['month_abs']=[(e[0]-yearmin)*12+e[1] for e in df_index.year_month]
    coun_Ncells_time=df_index[['month_abs','coordbin']]
    
    # For each month (in absolute time), how many grid cells have data in them?
    # Let's group the data by month and count how  many coordsbin we have. 
    # Group by month: 
    coun_Ncells_time=df_index[['month_abs','coordbin']].set_index('coordbin').groupby('month_abs')
    # Access the dictionary of groups (groups are the keys, coords are the values):
    dic_month_Ncells=coun_Ncells_time.groups
    # For each month, let's count how many cells are filled.
    lis_coords_Nposts=[]
    for month in list(dic_month_Ncells.keys()):
        coord=dic_month_Ncells[month]
        # We want to keep only one occurence for each cell grid.
        lis_coords_Nposts.append((month, len(set(coord))))
    date_Ncells=[abs_to_yearmonth(m[0],yearmin) for m in lis_coords_Nposts]
    return date_Ncells, lis_coords_Nposts



def rem_emj(unicode_string,keys):
    """Removes words found in keys from unicode_string.
    
    Works to remove emojis by setting keys to emoji.unicode_codes.UNICODE_EMOJI.keys()"""
    #This works but it takes forever
    keys=list(emoji.unicode_codes.UNICODE_EMOJI.keys())
    ret=''
    for i in unicode_string:
        if not i in keys:
            ret+=i
    return ret

def no_punct(text):
    return ''.join(list(filter(lambda x : x not in string.punctuation, text)))

def text_processing(df_new):
    """cleans the text of the df_new pandas dataset passed in argument.
    
    The dataset has to have a column 'text' where the text is stored.
    The function text_processing removes hyperlinks, quotes, tags, tickers, numbers,
    hashtags and punctuation.
    Some emojis are also removed.
    """
    emoji_pattern = re.compile("["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    df_new=df_new[df_new['text'].notnull()]
    #Purge newline characters
    df_new['text']=df_new.text.replace('\n',' ',regex=True)
    #Convert all text in unicode format
    #if not type(df_new['text'].iloc[0])==str:
    #    df_new['text']=df_new['text'].map(lambda x: x.decode('utf-8'))
    #remove hyperlinks
    df_new['text_new'] = df_new.text.map(lambda x: re.sub(r'http\S+', ' ', x))
    #remove quotes
    df_new.text_new = df_new.text_new.map(lambda x: re.sub(r'&amp;quot;|&amp;amp', ' ', x))
    #Remove tags
    df_new.text_new = df_new.text_new.map(lambda x: re.sub(r'@[a-zA-Z0-9]*', ' ', x))
    #Remove tickers
    df_new.text_new= df_new.text_new.map(lambda x: re.sub(r'\$[a-zA-Z0-9]*', ' ', x))
    #Remove numbers
    df_new.text_new=df_new.text_new.map(lambda x:  re.sub(r'[0-9]*','',x))
    #Remove #
    df_new.text_new=df_new.text_new.map(lambda x:  re.sub(r'#*','',x))
    # Set only a space between each word
    df_new.text_new=df_new.text_new.map(lambda x: ' '.join([a for a in x.split(' ') if a!='']))
    #Remove punctuation
    df_new.text_new=df_new.text_new.apply(no_punct)
    #encoding back in strings
    df_new.text_new = df_new.text_new.map(lambda x: x.encode('utf-8'))
    #Clean the posts without text and with only a space as content
    df_new=df_new[df_new['text_new'].notnull()]
    df_new=df_new[df_new['text_new'] != '']
    return df_new

def build_grid(df_new, step_m):
    """Defines the bins and binning functions, for squared bins of side step_m in km.
    
    """
    lat1=min(df_new.lat)
    lat2=max(df_new.lat)
    lon1=min(df_new.lon)
    lon2=max(df_new.lon)
    step_lon=step_m/(40000*math.cos((lat1+lat2)*math.pi/360)/360)
    step_lat=step_m/(40000./360.)
    # Define the repartition functions...
    to_bin_lon = lambda x: np.floor(x / step_lon) * step_lon
    to_bin_lat = lambda x: np.floor(x / step_lat) * step_lat
    return lat1,lat2,step_lat,lon1,lon2,step_lon,to_bin_lon,to_bin_lat

def add_grid_todf(df_new,to_bin_lat,to_bin_lon,step_lat,step_lon):
    df_new["latbin"] = df_new.lat.map(to_bin_lat)
    df_new["lonbin"] = df_new.lon.map(to_bin_lon)
    #Form the bins
    #groups = df_new.groupby(["latbin", "lonbin"])
    #compute the number of rows and columns
    number_lats=(max(df_new['latbin'])-min(df_new['latbin']))/step_lat+1
    number_lon=(max(df_new['lonbin'])-min(df_new['lonbin']))/step_lon+1
    return number_lats,number_lon,df_new
def name_id_degeneracy(df_new):
    """Computes degeneracy between names and places id the passed pandas dataset.
    
    """
    temp_df=df_new.copy()
    # Let's make a list of coordinates associated to each place and store them in a pd.Series
    A=temp_df.groupby('name')['coords'].apply(lambda x : x.tolist())
    B=temp_df.groupby('coords')['name'].apply(lambda x : x.tolist())
    # Idem but with the bins associated to each place
    Abin=temp_df.groupby('name')['coordsbin'].apply(lambda x : x.tolist())
    Bbin=temp_df.groupby('coordsbin')['name'].apply(lambda x : x.tolist())

    degenerate_datapoints=0
    Degenerate_loc=[]
    Degenerate_names=[]
    Degenerate_freq=[]
    # For every place in A
    for a in A.index:
        # if we have more than one coordinate for this place
        if len(set(A[a])) != 1:
            degenerate_datapoints+=len(A[a])
            #Store the place name, coordinates and relative frequencies in 3 lists of lists.
            Degenerate_loc.append(list(set(A[a])))
            Degenerate_names.append(a)
            Degenerate_freq.append([(key,len(list(group))) for key, group in groupby(A[a])])
    print('BEFORE BINNING:')
    print((degenerate_datapoints, ' degenerate datapoints in the dataset.'))
    print(100*degenerate_datapoints/df_new.shape[0], '% of degenerate data')

    degenerate_datapoints=0
    Degenerate_locbin=[]
    Degenerate_namesbin=[]
    # For every place in A
    for a in Abin.index:
        # if we have more than one bin for this place
        if len(set(Abin[a])) != 1:
            #Store the place name and bin 2 lists of lists.
            degenerate_datapoints+=len(Abin[a])
            Degenerate_locbin.append(list(set(Abin[a])))
            Degenerate_namesbin.append(a)
    print('AFTER BINNING:')
    print(degenerate_datapoints, ' degenerate datapoints in the dataset.')
    print(100*degenerate_datapoints/df_new.shape[0], '% of degenerate data')
    return Degenerate_loc, Degenerate_names, Degenerate_locbin, Degenerate_namesbin,Degenerate_freq

def pairwise_distances(Degenerate_loc,Degenerate_names):
    # Now compute the pairwise distance to have an idea of the area associated to those places.
    distances=[]
    Degloc_copy=np.copy(Degenerate_loc)
    for d in Degloc_copy:
        dist_d=[]
        e=list(np.copy(d))
        while e:
            #take the first point as reference to compute its distance to the other ones
            ref=e.pop()
            for point in e:
                try:
                    dist_d.append(geopy.distance.distance(ref,point).km)
                except IndexError:
                    pass
        distances.append(dist_d)

    Deg_names_dist=list(zip(Degenerate_names,distances))
    return Deg_names_dist

def deg_places_rel_freq(Degenerate_freq,step_lat,sep_lon,distances,Degenerate_names,freq_thr,dist_thr):
    """Prints a list of places with degenerated GPS coord if the distibution is uneven.
    
    If the relative frequency of a GPS coordinate is higher than freq_thr, the function will
    print the name of the place, the GPS coordinates and the correspondant relative frequencies
    if the typical pairwise distance associated to those coordinates is bigger than dist_thr.
    """
    # Find the relative frequencies of the different locations associated 
    # to each degenerated place
    Resp_freq=[[i[1] for i in e] for e in Degenerate_freq]
    Resp_freq=[[float(i)/sum(e) for i in e] for e in Resp_freq]
    # zip(Resp_freq,distances)
    bin_dist=geopy.distance.distance((45.108768700000006, 7.6410829),(45.108768700000006+step_lat,step_lon+ 7.6410829)).km
    print('The diagonal of the grid unit cell is typically ',bin_dist,' km.')
    print(len([d for d in distances if np.mean(d)>bin_dist]), ' places have typical pairwise distances bigger than this distance.')
    # List of the degenerated places with their relative frequencies, names 
    # and typical pairwisedistance.

    C=list(zip(Resp_freq,Degenerate_names,distances))
    for c in C:
        # If one coordinate has a frequency much higher than the others
        if max(c[0])>freq_thr:
            # And if the average pairwise distance is bigger than a threshold
            if np.mean(c[2])>dist_thr:
                print(c[1], Degenerate_loc[Degenerate_names.index(c[1])], np.mean(c[2]))
                
                
                
def first_apparition(df_new):
    # Plot the N_posts for each place as a function of their earliest occurence in the dataset. 
    temp_df=df_new.copy()
    # group data by name, then compute the size of each group and store that in a dict
    dict_places_n=temp_df.groupby('location_id').count()['text'].to_dict()
    # Same but instead of the size compute the minimum date in each group (earliest occurence)
    dict_places_fdate=temp_df.groupby('location_id').apply(lambda x : x['date'].min()).to_dict()

    N_fdate=[]
    for k in list(dict_places_n.keys()):
        N_fdate.append((dict_places_n[k],dict_places_fdate[k],k))
    return N_fdate,dict_places_n

def plot_time_series(Places_to_get,dict_coord_name,df_new,filename,save):
    """Plots the time series for places in Places_to_get.
    
    Saves the pdf files in a folder called 'filename' without / at the end.
    """
    time_series=df_new.copy()
    # let's plot the time series for each of those "places to get".
    time_series["date"] = time_series["date"].astype("datetime64")

    if save and (not filename in os.listdir('.')):
        os.mkdir(filename)

    for p in Places_to_get:
        #PlaceName is the name of the place
        PlaceName= dict_coord_name[p[0]]
        print(PlaceName, p[1])
        # Keep only the data where the field name matches PlaceName
        time_series=df_new.loc[df_new['name'] == PlaceName]
        time_series.set_index('date', drop=False, inplace=True)
        #Resample data on a monthly 'M' frequency and plot it with matplotlib
        ax = time_series.text.resample('M').count().plot(x_compat=True)
        #The hashtag Torino Centro lasts for only a year, it is not enough to have a tick every 6 moths
        if PlaceName=='Torino Centro':
        # elsewhere data spans over several years, so a tick in Jan and one in Jul is enough.
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
        ax.xaxis.set_label('timeline')
        ax.yaxis.set_label(r'N_{occurences}')
        ax.set_title(PlaceName)
        if save:
            plt.savefig(filename+'/'+PlaceName.replace(' ', '_')+'.pdf')
        plt.show()
    
def indexed_dataframe(df_new,step_lon,step_lat,number_lats):
    """returns 2 dfs, resp. indexing GPS on the grid matrix and months on an absolute timeline.
    
    """
    # Hotspot = coordinates associated with several posts.
    # How many different hotspots do we have? 
    df_new_index_cells=df_new.copy()
    # group the posts by coordinates and agglomerate the text in each of them
    df_new_index_cells['text'] = df_new_index_cells.groupby(['coords'])['text'].transform(lambda x: ','.join(x))
    # Keep only one occurence for each hotspot
    df_new_nodup_cells=df_new_index_cells[['latbin','lonbin','coords']].drop_duplicates()
    print('How many spots?', df_new_nodup_cells.shape[0])

    df_new_nodup_cells['row']=(max(df_new_nodup_cells['lonbin'])-df_new_nodup_cells['lonbin'])/step_lon
    df_new_nodup_cells['column']=(df_new_nodup_cells['latbin']-min(df_new_nodup_cells['latbin']))/step_lat
    df_new_nodup_cells['(r,c)'] = list(zip(df_new_nodup_cells['row'],df_new_nodup_cells['column']))
    df_new_nodup_cells['n'] = (df_new_nodup_cells['row'])*number_lats+df_new_nodup_cells['column']

    #And index it by coordinates
    df_new_nodup2_cells=df_new_nodup_cells.copy()
    df_new_nodup2_cells=df_new_nodup2_cells.set_index(['coords'])
    return df_new_index_cells, df_new_nodup2_cells

def TimeFlags(start,stop,granularity,n_gran):
    flags=[start]
    if granularity=='months':
        T_delta=relativedelta(months=n_gran)
    elif granularity=='days':
        T_delta=relativedelta(days=n_gran)        
    while flags[-1]+T_delta<stop:
        flags.append(flags[-1]+T_delta)
    return flags


def dataframe_classified(slices,df_new_index):
    df_classified=pd.DataFrame()
    for flag in slices:
            #Select only the data corresponding to this month
            df_sliced= df_new_index[(df_new_index['date']>=flag[0]) & (df_new_index['date']<=flag[1])]
            gc.collect()
            #Group the data by bin and join the text of all the posts grouped this way
            print(flag[0].date(),flag[1].date(), len(set(df_sliced['text'].tolist())))
            try:
                df_sliced['text'] = df_sliced.groupby('coordsbin')['text'].transform(lambda x: ','.join(x))
            except ValueError as e:
                print(repr(e))
                continue
            #drop the potential duplicates
            # Some posts, at various dates, have the same text in them. We keep only the
            # first occurence and discard the next ones.
            df=(df_sliced[['text','coordsbin']].drop_duplicates())
            df.set_index('coordsbin', inplace=True)
            df.rename(columns={"text": slices.index(flag)},inplace=True)
            df_classified=df_classified.merge(df, how='outer', left_index=True,right_index=True)
    df_classified.fillna(value='',inplace=True)
    series_list = [df_classified[c] for c in df_classified.columns[:-1]]
    # concatenate:
    df_classified['result'] = [s.strip() for s in series_list[0].str.cat(series_list[1:], sep=' ')]
    return df_classified

def dataframe_sampled(df,start):
    #How many tweets this month?
    list_reviews_rest=df[start].tolist()
    list_reviews_rest = [a.split(',') for a in list_reviews_rest if a!='']
    threshold=0
    for b in list_reviews_rest:
        threshold+=len(b)
    print('The threshold is', threshold)
    
    for col in df.columns:
        Text=df[col].tolist()
        #Split the posts: how many of them do we have in each bin?
        list_reviews_rest = [len(a.split(',')) for a in Text]
        if threshold<sum(list_reviews_rest):
            #Which bins have info?
            list_indices_info=[Text.index(a) for a in Text if a!='']
            #If we have more posts this month than in the reference month
            #On a bin with info...
            choices=[np.random.choice(list_indices_info) for i in range(threshold)]
            #...Pick a tweet among those there.
            choices =[(e,np.random.randint(0,list_reviews_rest[e])) for e in choices]
            This_Month=['' for i in Text]
            for tweet in choices:
                This_Month[tweet[0]]+=Text[tweet[0]].split(',')[tweet[1]]+','
            #new_df = pd.DataFrame({col: This_Month})
            new_column = pd.Series(This_Month, name=col, index=df.index)
            df.update(new_column)
    return df

def abs_to_yearmonth(month_abs,list_years):
    year_month=str(min(list_years)+int((month_abs-month_abs%12)/12))+' '+str(month_abs%12+1)
    return pd.Timestamp(year_month)



def NTF_sampling(sampling,df_classified,flags,start,stop,path,vectorizer_new=0,vectorizer_s=0,n_topics=10,n_features=1000,matlab=False,monuments=False):
    # Feed the vectorizer with all the words in the dataset. Counts is the tweet/term matrix.
    # fit_transform: fit first (build the features list with the relevant words)
    # then transform: build the tweet/term matrix with the relevant tokens.
    if not vectorizer_new:
        print('No vectorizer defined. Returning None')
        return None
    if matlab:
        name_matlab=path+'matlab/TorInst{}Matr'.format(n_features)
    Coord_CRS_global=[]
    Data_CRS_global=[]
    Ncells=[]
    snapshots=df_classified.columns.tolist()[0:-1]
    if start-stop >=0 or stop==0:
        print('incorrect start and/or stop dates. performing NTF on whole passed dataset')
        stop=min(len(flags), len(snapshots))
        
    #For every snapshot taken
    ct=0
    for month in snapshots[start:stop]:
        ct+=1
        print(flags[snapshots.index(month)])
        This_Month=df_classified[month].tolist()
            #print len(list_reviews_rest), 'tagged cells for ', year_month

        # Learn the vocabulary dictionary and return term-document matrix.
        print(len(This_Month))
        counts = vectorizer_new.transform(This_Month)
        #Transform a count matrix to a normalized tf-idf representation. 
        #(i.e terms with frequencies too hi or lo are removed)
        # Weights are indexed by (postID, term): weight
        tfidf = TfidfTransformer().fit_transform(counts)
        if matlab:
            savemat(name_matlab+str(ct), {'tfidf':tfidf})
        #print 'tfidf done:'
        #print tfidf
        C,D=read_CRS_totensor(tfidf,n_features,snapshots.index(month)-start)
        #print 'C,D'#, C,D
        Coord_CRS_global.append(C)
        Data_CRS_global.append(D)
        
    triples=[]
    triples_data=[]
    #For every month in the timeline
    for i in range(0,len(Coord_CRS_global)):
        c=Coord_CRS_global[i]
        # For every post in this month
        for e in c:
            #Add the non-zero elements coordinates
            triples.append(e)

    for d in Data_CRS_global:
        for e in d:
            triples_data.append(e)
    triples=[list(i) for i in triples]
    try:
#        maxNcells=max([e[0] for e in triples])
        maxNcells=len(df_classified[start])
    except ValueError:
        print('no non-zero element. returning None')
        print(triples)
        
    # Build a sktensor, which is ncp friendly. The dimensions have to be
    # N_bins x n_features x N_months.
    # N_months = len(Nposts) e.g, or len(Coord_CRS_global)
    # N_posts_total=sum(Nposts)
    X = sktensor.sptensor(tuple(list(np.asarray(triples).T)), triples_data, shape=(maxNcells, n_features, min(stop-start,df_classified.shape[1])))
    X_approx_ks = ncp_py3.nonnegative_tensor_factorization(X, n_topics, method='anls_bpp')
    A = X_approx_ks.U[0]
    B = X_approx_ks.U[1]
    C = X_approx_ks.U[2]
    lambdas = X_approx_ks.lmbda

    voc_vector={k:v for v,k in vectorizer_s.vocabulary_.items()}
    voc_serie=pd.Series(voc_vector)
    TermVectors=[]
    TermVectorsIndex=[]
    for row in B.T:
        row=list(row)
        row = [(r,row.index(r)) for r in sorted(row)[::-1]]
        TermVectors.append(set([voc_vector[e[1]] for e in row]))
        TermVectorsIndex.append([(voc_vector[e[1]],e[0]) for e in row])
    for i in range(0,len(TermVectorsIndex)):
        TermVectorsIndex[i].sort(key=lambda tup:tup[1])
        TermVectorsIndex[i]=TermVectorsIndex[i][::-1]
    return A,B,C,TermVectorsIndex,TermVectors,lambdas


def geo_heatmap(Coordinates, Nposts, save=False, filename='hm'):
    """Draws a folium heatmap based the histogram of the Coordinates variable. 
    
    Saves it if save
    """
    x = [C[0] for C in Coordinates]
    y = [C[1] for C in Coordinates]

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=1000)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    hmap = folium.Map(location=[np.mean(x), np.mean(y)], zoom_start=10, )

    hm_wide = HeatMap( list(zip(x, y, heatmap)),
                       min_opacity=0.2,
                       max_val=max(Nposts),
                       radius=17, blur=15, 
                       max_zoom=1, 
                     )
    hmap.add_child(hm_wide)
    if save:
        if filename=='hm':
            print('Warning: no name specified. heatmap saved as Heatmaps/hm.html')
        if 'Heatmaps' not in os.listdir('.'):
            try:
                os.mkdir('Heatmaps')
            except OSError:
                pass
            if message !='':
                f=file.open('Heatmaps/read_me','w+')
                f.write(message)
                f.close()
        hmap.save('Heatmaps/'+filename+'.html')
        
def geo_topic(row,Coordinates,index,topic='',save=False,folder='heatmaps'):
    if topic=='':
        print("Warning; topic undefined. saving the file under folder/index.html")
        topic=str(index)
    x=sorted(set([Cd[0] for Cd in Coordinates]))
    y=sorted(set([Cd[1] for Cd in Coordinates]))
    z=[(Coordinates[i], row[i]) for i in range(0,len(row))]
    hem=np.zeros((len(x),len(y)))
    for e in z:
        hem[x.index(e[0][0])][y.index(e[0][1])]=e[1]
    hmap = folium.Map(location=[np.mean(x), np.mean(y)], zoom_start=10, )

    hm_wide = HeatMap( list(zip([C[0] for C in Coordinates],[C[1] for C in Coordinates],row)),
                       min_opacity=0,
                           max_val=max(row),
                       radius=15, blur=0, 
                       max_zoom=1, 
                     )
    hmap.add_child(hm_wide)
    hmap.render()
    if save:
        if folder=='heatmaps':
            print('Warning: No folder name specified. The heatmaps will be saved in a created heatmaps folder')
        hmap.save(folder+'/heatmap'+topic+'_norm.html')
        
def file_tree(word2vec,seed_file,seed,n_features,sampling,flags,start,n_topics,monuments):
    if word2vec:
        df_word2vec=pd.read_pickle('PKL_files/'+seed_file+"/"+seed+'/df_word2vec.pkl')
        if '{}K_word2vec'.format(str(n_features/1000)) not in os.listdir('Figures/Heatmaps'):
            try:
                os.mkdir('Figures/Heatmaps/{}K_word2vec'.format(str(n_features/1000)))
            except OSError:
                pass
        if '{}K_word2vec'.format(str(n_features/1000)) not in os.listdir('PKL_files/results'):
            try: 
                os.mkdir('PKL_files/results/{}K_word2vec'.format(str(n_features/1000)))
            except OSError:
                pass

        if sampling:
            msg='Made on {}, sampled from {}. {} TV. {} words dictionary. Word2vec, seed {}'.format(datetime.now().date(), flags[start], n_topics, (n_features),seed)
            foldername1='{}K_word2vec/Heatmaps_{}_{}_spl_word2vec'.format(str(n_features/1000), datetime.now().date(), n_topics)
            foldername2='{}K_word2vec/NTF_{}_{}_spl_word2vec'.format(str(n_features/1000), datetime.now().date(), n_topics)
        else:
            msg='Made {}, no sampling. {} TV. {} words dictionary.  Word2vec, seed {}'.format(datetime.now().date(), n_topics, (n_features),seed)
            foldername1='{}K_word2vec/Heatmaps_{}_{}_word2vec'.format(str(n_features/1000), datetime.now().date(), n_topics)
            foldername2='{}K_word2vec/NTF_{}_{}_word2vec'.format(str(n_features/1000), datetime.now().date(), n_topics)

        if monuments:
            all_stopwords=all_stopwords.union(words_monuments)
            msg+=' No monuments.'
            foldername1+='_noMon'
            foldername2+='_noMon'
        else:
            all_stopwords=all_stopwords

    else:
        if '{}K'.format(str(n_features/1000)) not in os.listdir('Figures/Heatmaps'):
            try:
                os.mkdir('Figures/Heatmaps/{}K'.format(str(n_features/1000)))
            except OSError:
                pass
        if '{}K'.format(str(n_features/1000)) not in os.listdir('PKL_files/results'):
            try: 
                os.mkdir('PKL_files/results/{}K'.format(str(n_features/1000)))
            except OSError:
                pass

        if sampling:
            msg='Made on {}, sampled from {}. {} TV. {} words dictionary.'.format(datetime.now().date(), flags[start], n_topics, (n_features))
            foldername1='{}K/Heatmaps_{}_{}_spl'.format(str(n_features/1000), datetime.now().date(), n_topics)
            foldername2='{}K/NTF_{}_{}_spl'.format(str(n_features/1000), datetime.now().date(), n_topics)
        else:
            msg='Made on the 25/01, no sampling. {} TV. {} words dictionary.'.format(n_topics, (n_features))
            foldername1='{}K/Heatmaps_{}_{}'.format(str(n_features/1000), datetime.now().date(), n_topics)
            foldername2='{}K/NTF_{}_{}'.format(str(n_features/1000), datetime.now().date(), n_topics)

        if monuments:
            all_stopwords=all_stopwords.union(words_monuments)
            msg+=' No monuments.'
            foldername1+='_noMon'
            foldername2+='_noMon'
        else:
            all_stopwords=all_stopwords
    return foldername1, foldername2, msg, all_stopwords


def file_tree2(seeded,expanded,n_features,sampling,flags,start,n_topics,monuments,n_posts,seed='noseed'):
    run_id='_'.join([str(datetime.now().date()),str(n_topics),str(n_features)])
    if monuments:
        run_id+='_noMon'
    if sampling:
        run_id+='_spl'
    if seeded:
        run_id+='_{}'.format(seed)
    if expanded:
        run_id+='_word2vec'
    try:
        print(1)
        os.mkdir('Results/'+run_id)
    except OSError:
        #This file already exists. let's build a second folder
        indic=2
        while(True):
            #Try to make the new folder with the #2 added at the end
            try:
                os.mkdir('Results/'+run_id+'_#{}'.format(indic))
            except OSError:
                #If it already exists, try with #3...
                indic+=1
                print(indic)
                continue
            #Until it is done. exit the loop
            break
        run_id+='_#{}'.format(indic)
        pass
    
    os.chdir('Results/'+run_id)
    os.mkdir('Heatmaps')
    os.mkdir('t_series')
    os.mkdir('matlab')
    os.mkdir('pickle')
    f=open('read_me.txt','w+')
    f.write('File made on the {}'.format(datetime.now().date())+'\n')
    f.write('{} Term vectors asked'.format(n_topics)+'\n')
    f.write('{} words in the vocabulary'.format(n_features)+'\n')
    f.write('{} posts in the dataset'.format(n_posts)+'\n')
    
    if sampling:
        f.write('sampled from month {} (i.e {})'.format(start, abs_to_yearmonth(start, [2010]))+'\n')
    if monuments:
        f.write('No monuments \n')
    if seeded:
        f.write('Sampled with seed '+seed+'\n')
    if expanded:
        f.write('Expanded using word2vec \n')
    f.close()
    os.chdir('..')
    os.chdir('..')
    return 'Results/'+run_id+'/'

def stopwords(monuments, all_stopwords=all_stopwords, words_monuments=words_monuments):
    if monuments:
        all_stopwords=all_stopwords.union(words_monuments)
    else:
        all_stopwords=all_stopwords
    return all_stopwords

def seeding(A,seed):
    return [a for a in A if seed in a]

def df_seeding(df_classified, seed):
    df=df_classified.copy()
    Tot=0
    All_seeded_text=[]
    for column in df_classified.columns:
        Text=df_classified[column].tolist()
        Text=[e.split(',') for e in Text]
        New_text=[seeding(e, seed) for e in Text]
        All_seeded_text.append(New_text)
        new_column = pd.Series([string.join(a,', ') for a in New_text], name=column, index=df.index)
        df.update(new_column)
    for month in All_seeded_text:
        for Bin in month:
            if Bin!=[]:
                Tot+=len(Bin)
    #Tot takes into account each column + the column results which is a sum of all other columns.
    #Therefore it is twice the number of tweets in the corpus.
    print(seed, Tot/2)
    return df

def expanded_seeding(A,expanded):
    return [a for a in A if ((set(a.split(' ')) & set(expanded))!=set())]

def df_expanded_seeding(df_classified, seed, model=None):
    if not model:
        os.listdir('.')
        model= gensim.models.Word2Vec.load('models/model_I5_win10')
    else:
        pass
    exp_seeds=[e[0] for e in model.wv.most_similar(positive=seed, topn=5)]
    print(seed, exp_seeds)
    df=df_classified.copy()
    Tot=0
    All_seeded_text=0
    for column in df.columns:
        Text=df[column].tolist()
        if (set(string.join(Text, ' ').split(' ')) & set(exp_seeds) == set([])):
            new_column = pd.Series(['' for a in Text], name=column, index=df.index)
            pass
        else:
            Text=[e.split(',') for e in Text]
            New_text=[expanded_seeding(e, exp_seeds) for e in Text]
            All_seeded_text+=sum([len([n for n in N if n!='']) for N in New_text])
            new_column = pd.Series([string.join(a,', ') for a in New_text], name=column, index=df.index)
        df.update(new_column)
    #All_seeded_text takes into account each column + the column results which is a sum of all other columns.
    #Therefore it is twice the number of tweets in the corpus.
    print(seed, All_seeded_text/2)
    return df

def train_word2vec(I, workers, window, save=True, override=False):
    if 'model_I{}_win{}'.format(I,window) in os.listdir('models') and not override:
        print('This model has already been computed; returning already existing version.')
        return model.load()
    model = gensim.models.Word2Vec(sentences,iter=I,workers=workers,window=window)
    if save:
        model.save('/models/model_I{}_win{}'.format(I,window))
    return model

def language_counter(all_posts):
    languages_counter={}
    for a in set(all_posts):
        L=language_detector(a.decode('utf-8'))
        if L in list(languages_counter.keys()):
            languages_counter[L]+=1
        else:
            languages_counter[L]=1
    return language_counter