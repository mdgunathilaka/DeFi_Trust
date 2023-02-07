import praw
from datetime import datetime
import pandas as pd

class Reddit_API:
    def __init__(self):
        self.user_agent = ""
        self.reddit = praw.Reddit(
            client_id="",
            client_secret="",
            user_agent=self.user_agent
        )
        #subreddits to collect data from
        self.subr = "Cryptocurrency+CryptoMarkets+BitcoinBeginners+CryptoTechnology+CryptoCurrencies+DeFi+binance+Ethereum"

    def collect_reddit(self, symbol, name):
        reviews = []
        n = 0
        for submission in self.reddit.subreddit(self.subr).search(symbol or name, sort="new", limit=10):
            # print(str(datetime.fromtimestamp(submission.created_utc)))
            #print(submission.title)
            # print(submission.selftext)
            n += 1
            if len(reviews)>=100:
                break
            if str(datetime.fromtimestamp(submission.created_utc))[:3] == "2017":
                print("2017")
                break
            reviews.append([str(datetime.fromtimestamp(submission.created_utc)), submission.title])
            if submission.selftext!="":
                reviews.append([str(datetime.fromtimestamp(submission.created_utc)), submission.selftext])
            submission.comment_sort = "new"
            submission.comments.replace_more(limit=0)  # flatten tree
            #print(n)
            print(f"collecting {symbol}")
            for top_level_comment in submission.comments:
                reviews.append([str(datetime.fromtimestamp(top_level_comment.created_utc)),
                                submission.title + " " + top_level_comment.body])
                if len(reviews) >= 100:
                    break
        #print(len(reviews))
        reviews.sort()
        return reviews
'''
def initialize_api():
    user_agent = "fypcollect 1.0 by /u/mdgunathilaka"
    reddit = praw.Reddit(
        client_id = "my6X4q4K51jqIszvtaPC-w",
        client_secret = "aw3_x3fl0CtYz5Le9dEI9rzyerSnUw",
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

def collect_reddit(token, name):
    reviews = []
    n = 0
    for submission in reddit.subreddit(subr).search(token or name, sort="new", limit=2):
        # print(str(datetime.fromtimestamp(submission.created_utc)))
        #print(submission.title)
        # print(submission.selftext)
        n += 1
        reviews.append([str(datetime.fromtimestamp(submission.created_utc)), submission.title])
        reviews.append([str(datetime.fromtimestamp(submission.created_utc)), submission.selftext])
        submission.comment_sort = "hot"
        submission.comments.replace_more(limit=0)  # flatten tree
        print(n)
        if str(datetime.fromtimestamp(submission.created_utc))[:3] == "2016":
            print("2016")
            break
        for top_level_comment in submission.comments:
            reviews.append([str(datetime.fromtimestamp(top_level_comment.created_utc)),
                            submission.title + " " + top_level_comment.body])
    print(len(reviews))
    reviews.sort()
    for i in reviews:
        print(i)

#collect_reddit('RCLE', 'Ross Campbell Legal Engineering')
collect_reddit('SBOTS', 'Scuttle Bots | t.me/ScuttleBots')
'''