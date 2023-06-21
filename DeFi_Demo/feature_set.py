import pandas as pd
from collections import defaultdict
import numpy as np

#answered in stackoverflow by https://stackoverflow.com/users/562769/martin-thoma
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    #print(f"total:{total}, np mean:{np.mean(x)}")
    return total / (len(x)**2 * np.mean(x))

def get_features(token, transfers):
    #transfers = pd.read_csv("./scam_event_series/"+token)
    N = len(transfers)//10
    if N>108:
        N=108

    df = pd.DataFrame(columns=["block_difference", "volume", "minted", "burnt", "n_unique_addresses", "gini_coefficient","avg_gas_price"])
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
            gas_total += transfers["gas_price"][j]
            volume += value_
            # mints
            if from_ == "0x0000000000000000000000000000000000000000":
                minted += value_
            # burns
            if to_ == "0x0000000000000000000000000000000000000000":
                burned += value_

        row.append(volume)
        row.append(minted)
        row.append(burned)

        n_unique = len(address_dict)
        row.append(n_unique)
        if volume == 0:
            row.append(1 / n_unique)
        else:
            # print("gini started")
            gini_c = gini(np.array(list(address_dict.values())))
            row.append(gini_c)
            # print("gini ended")

        # row.append(sum(gas_price_s)/10)
        row.append(gas_total / 10)

        df.loc[i] = row

    #df.to_csv("./ankh/first/eth_feature/" + token)
    #df.to_csv("./ankh/full/eth_feature/" + token)
    df.to_csv("feature_df.csv")
    df = pd.read_csv("feature_df.csv")
    cols = list(df.columns)
    df = df.fillna(0)
    df[cols[-1]] = df[cols[-1]] / max(df[cols[-1]])

    for col in cols[2:5]:
        maxv = max(df[col])
        if maxv != 0:
            df[col] = df[col] / maxv
    df = df.fillna(0)
    feature_data = []
    for index, row in df.iterrows():
        # print(list(row)[1:])
        feature_data.append(list(row)[1:])
    n = len(feature_data)

    if n < 108:
        feature_data = feature_data + [[0, 0, 0, 0, 0, 0, 0]] * (108 - n)

    return feature_data