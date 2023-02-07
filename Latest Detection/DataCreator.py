from utils.event_collector import *
from utils.reddit_collector import *
import pandas as pd
import os


#collect all scam transactions and reddit
eth_api = ETH_API()
reddit_api = Reddit_API()
#f = open("collected_scam.txt", "w")
#f.write("token symbol name")
f = open("collected_scam.txt", "a") #2nd time collecting
Label_df = pd.read_csv('labeled_list.csv')
scam_df = Label_df.loc[Label_df['label']==0]
scam_df = scam_df.iloc[26404:] #2nd time collecting from the last point
i = 26404 #previous collected
j = 2565 #previous collected
for address in scam_df['token_address']:
    i+=1
    print(i)
    df = eth_api.get_transfers(address, './data/eth_log/scam')
    if len(df)>100:
        try:
            name, symbol = eth_api.get_token_name(address)
            print(f"{symbol} transfers collected")
            reviews = reddit_api.collect_reddit(symbol, name)
            if len(reviews)>0:
                reviews_df = pd.DataFrame(reviews, columns=["timestamp", "review"])
                reviews_df["review"].to_csv("./data/reddit_data/scam/"+address+".csv")
                j += 1
                print(f"{j} {address} {symbol} {name}")
                f.write("\n"+address+" "+symbol+" "+name)
        except Exception as err:
            #some token have name and symbol as bytes32. By changing the ABI those tokens also can be collected
            print(f"Exception occured: {err}")
            print(f"abnormal address: {address}")
f.close()
print("scam done")