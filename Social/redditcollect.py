import praw
from datetime import datetime
import pandas as pd

def initialize_api():
    user_agent = ""
    reddit = praw.Reddit(
        client_id = "",
        client_secret = "",
        user_agent = user_agent
    )
    print("done")
    return reddit

def get_subreddit(reddit, name):
    subreddit = reddit.subreddit(name)
    return subreddit

subs = ["Cryptocurrency", "CryptoMarkets", "BitcoinBeginners", "CryptoTechnology", "CryptoCurrencies", "DeFi", "binance", "Ethereum"]
subr = "Cryptocurrency+CryptoMarkets+BitcoinBeginners+CryptoTechnology+CryptoCurrencies+DeFi+binance+Ethereum"

reddit = initialize_api()
reviews = []
for submission in reddit.subreddit(subr).search('FTX', sort="hot", limit=2):
    #print(str(datetime.fromtimestamp(submission.created_utc)))
    #print(submission.title)
    #print(submission.selftext)
    reviews.append([str(datetime.fromtimestamp(submission.created_utc)), submission.title])
    submission.comment_sort = "hot"
    submission.comments.replace_more(limit=0)  # flatten tree
    for top_level_comment in submission.comments:
        reviews.append([str(datetime.fromtimestamp(top_level_comment.created_utc)), top_level_comment.body])
'''
    comments = submission.comments.list()  # all comments
    if (len(comments)!=0):
        for comment in comments:
            #print(str(datetime.fromtimestamp(comment.created_utc)))
            #print(comment.body)
            reviews.append([str(datetime.fromtimestamp(comment.created_utc)), comment.body])
'''
print(len(reviews))
for i in range(20):
    print(reviews[i])
reviews.sort()
print("sorted")
for i in range(20):
    print(reviews[i])