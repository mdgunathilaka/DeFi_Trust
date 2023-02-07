import time

import pandas as pd
import numpy as np
from collections import defaultdict
from web3 import Web3, HTTPProvider
import os
import threading
import requests

INFURA_URL = "" #100,000 daily
GB_URL = "" #100,000 daily
web3 = Web3(HTTPProvider(INFURA_URL))

QUICKNODE_URL = "" #10mil monthly
web3_QN = Web3(HTTPProvider(QUICKNODE_URL))

ALCHEMY_URL = "" #300min monthly
web3_AL = Web3(HTTPProvider(ALCHEMY_URL))

BLOCKPI_URL = "" #100mil monthly #not archieve
web3_BP = Web3(HTTPProvider(BLOCKPI_URL))

TATUM_KEY = ""

#answered in stackoverflow by https://stackoverflow.com/users/562769/martin-thoma
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    #print(f"total:{total}, np mean:{np.mean(x)}")
    return total / (len(x)**2 * np.mean(x))

def get_gas(tx_id, arr, loc):
    #Infura
    gas = (web3.eth.get_transaction(tx_id).gasPrice) / (10 ** 18)
    arr[loc] = gas

def get_gas_tatum(tx_id, arr, loc):
    #Tatum
    url = "https://api.tatum.io/v3/ethereum/transaction/" + tx_id
    headers = {"x-api-key": TATUM_KEY}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        #print(data)
        gas = int(data['gasPrice']) / (10 ** 18)
    except:
        print("used infura")
        gas = (web3.eth.get_transaction(tx_id).gasPrice) / (10 ** 18)
    arr[loc] = gas

def get_gas_QN(tx_id, arr, loc):
    #Quicknode
    gas = (web3_QN.eth.get_transaction(tx_id).gasPrice) / (10 ** 18)
    arr[loc] = gas

def get_gas_AL(tx_id, arr, loc):
    #Alchemy
    gas = (web3_AL.eth.get_transaction(tx_id).gasPrice) / (10 ** 18)
    arr[loc] = gas

def get_gas_BP(tx_id, arr, loc):
    #Block PI
    try:
        gas = (web3_BP.eth.get_transaction(tx_id).gasPrice) / (10 ** 18)
    except:
        print("used infura")
        gas = (web3.eth.get_transaction(tx_id).gasPrice) / (10 ** 18)
    arr[loc] = gas

def get_features(token, transfers):
    #transfers = pd.read_csv("./scam_event_series/"+token)
    N = len(transfers)//10
    if N>108:
        N=108

    gas_prices = [0 for k in range(len(transfers))]
    thread_list = []

    #Infura
    # for t in range(len(transfers)):
    #     th = threading.Thread(target=get_gas, args=(transfers["transactionHash"][t], gas_prices, t,))
    #     thread_list.append(th)
    #     th.start()
    # for thr in thread_list:
    #     thr.join()

    #Tatum+QN+AL
    for t in range(len(transfers)):
        time.sleep(0.05)
        if(t%4==1):
            th = threading.Thread(target=get_gas_tatum, args=(transfers["transactionHash"][t], gas_prices, t,))
        elif(t%4==2):
            th = threading.Thread(target=get_gas_AL, args=(transfers["transactionHash"][t], gas_prices, t,))
        elif (t%4 == 3):
            th = threading.Thread(target=get_gas_BP, args=(transfers["transactionHash"][t], gas_prices, t,))
        else:
            th = threading.Thread(target=get_gas_QN, args=(transfers["transactionHash"][t], gas_prices, t,))
        thread_list.append(th)
        th.start()
    for thr in thread_list:
        thr.join()
    print("gas collected")

    df = pd.DataFrame(columns=["block_difference", "volume", "minted", "burnt", "n_unique_addresses", "gini_coefficient", "avg_gas_price"])
    for i in range(N):
        #print(i)
        row = []
        row.append(transfers["block_number"][(i+1)*10-1]-transfers["block_number"][i*10])
        volume = 0
        minted = 0
        burned = 0
        #gas_price_s = [0 for k in range(10)]
        address_dict = defaultdict(float)
        gas_total = 0
        #thread_list = []
        #rank = 0
        for j in range(i * 10, (i + 1) * 10):
            from_ = transfers["from"][j]
            to_ = transfers["to"][j]
            value_ = transfers["value"][j]
            address_dict[to_] += value_
            address_dict[from_] += value_

            #print("gas started")
            #gas_price = (web3.eth.get_transaction(transfers["transactionHash"][j]).gasPrice) / (10 ** 18)
            #gas_price_s += gas_price
            #print("gas collected")
            #th = threading.Thread(target=get_gas, args=(transfers["transactionHash"][j],gas_price_s,rank,))
            #thread_list.append(th)
            #th.start()
            #rank += 1
            gas_total += gas_prices[j]

            volume += value_
            # mints
            if from_ == "0x0000000000000000000000000000000000000000":
                minted += value_
            # burns
            if to_ == "0x0000000000000000000000000000000000000000":
                burned += value_
        #for t in thread_list:
        #    t.join()

        row.append(volume)
        row.append(minted)
        row.append(burned)

        n_unique = len(address_dict)
        row.append(n_unique)
        if volume==0:
            row.append(1/n_unique)
        else:
            #print("gini started")
            gini_c = gini(np.array(list(address_dict.values())))
            row.append(gini_c)
            #print("gini ended")

        #row.append(sum(gas_price_s)/10)
        row.append(gas_total / 10)


        df.loc[i] = row
    # print(df)
    # print(token)
    #df.to_csv("./data/eth_feature/test/" + token)
    #df.to_csv("./data/eth_feature/healthy/" + token)
    df.to_csv("./data/eth_feature/scam/" + token)
    return

if __name__ =="__main__":
    path = os.getcwd()
    #dir_list = os.listdir(path+"./data/reddit_data/test")
    #col_list = os.listdir(path + "./data/eth_feature/test_s")
    # dir_list = os.listdir(path + "./data/reddit_data/healthy/text")
    # col_list = os.listdir(path + "./data/eth_feature/healthy")
    dir_list = os.listdir(path + "./data/reddit_data/scam/text")
    col_list = os.listdir(path + "./data/eth_feature/scam")
    i = len(col_list)
    for item in dir_list:
        if (".csv" in item) and (item not in col_list):
            i += 1
            print(i)
            #df = pd.read_csv("./data/eth_log/healthy/"+item)
            df = pd.read_csv("./data/eth_log/scam/" + item)
            print(item)
            get_features(item, df)

    print("done")
