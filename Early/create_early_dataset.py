import numpy as np
import pandas as pd
import os
import gc
import json
from sklearn.preprocessing import MinMaxScaler

Xs = [] #shape = N, 48, 8
Labels_l = [] #shape = N

path = os.getcwd()
dir_list = os.listdir(path+"/scam_data_logs")

for item in dir_list:
    if ".csv" in item:
        df = pd.read_csv("./scam_data_logs/"+item)
        if df['max_share'][0] > 1:
            continue
        cols = list(df.columns)
        scaler = MinMaxScaler()
        df[cols[1:-2]] = scaler.fit_transform(df[cols[1:-2]])
        df = df.fillna(0)
        data = []
        for index, row in df.iterrows():
            #print(list(row)[1:])
            data.append(list(row)[1:])

        n = len(data)
        if n>=48:
            Xs.append(data[:48])
        else:
            data = data + [[0,0,0,0,0,0,0,0]]*(48-n)
            Xs.append(data)

        Labels_l.append(0)

print(f"scam token count = {len(Labels_l)}")

dir_list = os.listdir(path+"/healthy_data_logs")
for item in dir_list:
    if ".csv" in item:
        df = pd.read_csv("./healthy_data_logs/"+item)
        if df['max_share'][0] > 1:
            continue
        cols = list(df.columns)
        scaler = MinMaxScaler()
        df[cols[1:-2]] = scaler.fit_transform(df[cols[1:-2]])
        df = df.fillna(0)
        data = []
        for index, row in df.iterrows():
            #print(list(row)[1:])
            data.append(list(row)[1:])

        n = len(data)
        if n >= 48:
            Xs.append(data[:48])
        else:
            data = data + [[0, 0, 0, 0, 0, 0, 0, 0]] * (48 - n)
            Xs.append(data)

        Labels_l.append(1)
        #print(data)
        #print(Labels_l)
print(f"total token count = {len(Labels_l)}")

json_object = json.dumps(Xs)

# Writing to sample.json
with open("Xs.json", "w") as outfile:
    outfile.write(json_object)

Data = np.array(Xs)
print(Data.shape)
np.save("Data.npy",Data)

del Data
gc.collect()

Labels = np.array(Labels_l)
np.save("Labels.npy",Labels)
