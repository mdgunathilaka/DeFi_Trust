import pandas as pd
import networkx as nx
from collections import defaultdict

def get_features(token, transfers):
    #transfers = pd.read_csv("./scam_event_series/"+token)
    N = len(transfers)//10
    if N>48:
        N=48
    df = pd.DataFrame(columns=["block_difference", "volume", "supply", "mint_difference", "burns", "new_addresses", "cluster_coeffcient", "max_share"])
    total = 0
    G = nx.DiGraph()
    n = 0

    for i in range(N):
        #print(i)
        row = []
        row.append(transfers["block_number"][(i+1)*10-1]-transfers["block_number"][i*10])
        volume = 0
        minted = 0
        burned = 0
        mint_dict = defaultdict(float)
        for j in range(i*10,(i+1)*10):
            from_ = transfers["from"][j]
            to_ = transfers["to"][j]
            value_ = transfers["value"][j]

            volume += value_
            #mints
            if from_=="0x0000000000000000000000000000000000000000":
                mint_dict[to_] += value_
                minted += value_
            #burns
            if to_=="0x0000000000000000000000000000000000000000":
                burned += value_

            #add to graph
            try:
                G.add_edge(from_, to_, weight=(G.edges[from_, to_]['weight'] + value_))
            except KeyError:
                G.add_edge(from_, to_, weight=value_)
        row.append(volume)
        supply = minted - burned
        total += supply
        row.append(supply)

        if(mint_dict=={}):
            row.append(0)
        elif(len(mint_dict)==1):
            row.append(max(mint_dict.values()))
        else:
            row.append(max(mint_dict.values()) - min(mint_dict.values()))
        row.append(burned)

        new_n = len(G.nodes)
        #new addresses
        row.append(new_n - n)
        n = new_n

        cluster_coeffs = nx.average_clustering(G)
        row.append(cluster_coeffs)

        #calculate the token share of each account
        shares = []
        for add1 in list(G.nodes):
            balance = 0
            for add2 in list(G.predecessors(add1)):
                #print(ad)
                balance += G.edges[add2, add1]['weight']

            for add2 in list(G.neighbors(add1)):
                #print(ad)
                balance -= G.edges[add1, add2]['weight']
            try:
                shares.append(balance/total)
            except:
                print("token " + token + " has pre allocated tokens")
                return

        row.append(max(shares))

        df.loc[i] = row
    #print(df)
    #print(token)
    #df.to_csv("./scam_data_logs/"+token)
    df.to_csv("./healthy_data_logs/" + token)

import os
path = os.getcwd()
#dir_list = os.listdir(path+"/scam_event_series")
dir_list = os.listdir(path+"/healthy_event_series")
i = 0
for item in dir_list:
    print(i)
    i += 1
    if ".csv" in item:
        df = pd.read_csv("./healthy_event_series/"+item)
        if len(df)>50:
            print(item)
            get_features(item, df)

