import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

import tweepy as tw
import nltk
from nltk.corpus import stopwords
import re
import networkx
from textblob import TextBlob
import pickle

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

consumer_key= 'X'
consumer_secret= 'X'
access_token= 'X'
access_token_secret= 'X'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define a function to remove url from text
def remove_url(txt):
    """Replace URLs found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

print("#Collect tweets:")
tweets_GOOGL = tw.Cursor(api.search,q="AAPL",since="2020-04-03",until="2020-04-04",lang="en").items()
print("#Pick out relevant data:")
the_right_thing = [[tweet.created_at, TextBlob(remove_url(tweet.text)).sentiment.polarity] for tweet in tweets_GOOGL]
print("#Turn into dataframe:")
sentiment_df = pd.DataFrame(the_right_thing, columns=["created_at","polarity"])
print("#Save to .csv file:")
sentiment_df.to_csv("twitter_data_AAPL_2020_04_03.csv", encoding='utf-8', index=False)
