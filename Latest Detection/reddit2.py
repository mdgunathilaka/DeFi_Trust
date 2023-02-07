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