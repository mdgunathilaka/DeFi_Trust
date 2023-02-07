import pandas as pd
import networkx as nx
from collections import defaultdict

def get_features(token):
    transfers = pd.read_csv("../event_logs/"+token+".csv")
    N = len(transfers)//100
    df = pd.DataFrame(columns=["block_difference", "volume", "supply", "mint_difference", "burns", "new_addresses", "cluster_coeffcient", "max_share"])
    total = 0
    G = nx.DiGraph()
    n = 0

    for i in range(N):
        print(i)
        row = []
        row.append(transfers["block_number"][(i+1)*100-1]-transfers["block_number"][i*100])
        volume = 0
        minted = 0
        burned = 0
        mint_dict = defaultdict(float)
        for j in range(i*100,(i+1)*100):
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
            shares.append(balance/total)

        row.append(max(shares))

        df.loc[i] = row
    #print(df)
    df.to_csv("../time_series/"+token+".csv")

#token = "0x8DcFF4f1653f45cF418b0b3A5080A0fDCac577C8" #yai
token = "0x9b6443b0fB9C241A7fdAC375595cEa13e6B7807A" #rcc
token = "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984" #uni
get_features(token)