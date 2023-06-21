import praw
from datetime import datetime
import pandas as pd

class Reddit_API:
    def __init__(self):
        self.user_agent = "AGENT"
        self.reddit = praw.Reddit(
            client_id="YOUR ID",
            client_secret="YOUR SECRET",
            user_agent=self.user_agent
        )
        #subreddits to collect data from
        self.subr = "Cryptocurrency+CryptoMarkets+BitcoinBeginners+CryptoTechnology+CryptoCurrencies+DeFi+binance+Ethereum"

    def collect_reddit(self, symbol, name):
        reviews = []
        n = 0
        for submission in self.reddit.subreddit(self.subr).search(symbol or name, sort="best"):
            # print(str(datetime.fromtimestamp(submission.created_utc)))
            #print(submission.title)
            # print(submission.selftext)
            n += 1
            if str(datetime.fromtimestamp(submission.created_utc))[:4] == "2019":
                print("2019")
                break
            if len(reviews)>=100:
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
        reviews_df = pd.DataFrame(reviews, columns=["timestamp", "review"])
        return reviews_df
        #reviews_df.to_csv("./data/reddit_log/" + address + ".csv")