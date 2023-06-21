from collect_eth import *
from collect_reddit import *
from feature_set import *
import gradio as gr
from mlx import *
from PIL import Image

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import numpy as np

eth_api = ETH_API()
reddit_api = Reddit_API()

#Initializing DeFiTrust Model
class BinaryClassification(nn.Module):
  def __init__(self, embed_size, device):
    super(BinaryClassification, self).__init__()
    # Number of input features is embed_size. (108*7 + 100*5)
    self.layer_1 = nn.Linear(embed_size, 64)
    self.layer_2 = nn.Linear(64, 64)
    self.layer_out = nn.Linear(64, 1)

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.1)
    self.batchnorm1 = nn.BatchNorm1d(64)
    self.batchnorm2 = nn.BatchNorm1d(64)
    self.device = device
    #self.sigmoid = nn.Sigmoid()

  def forward(self, inputs):
    #print(inputs)
    #print(inputs.shape)
    x = self.relu(self.layer_1(inputs))
    x = self.batchnorm1(x)
    x = self.relu(self.layer_2(x))
    x = self.batchnorm2(x)
    x = self.dropout(x)
    x = self.layer_out(x)
    #print(x)
    #x = self.sigmoid(x)
    #if math.isnan (x[0][0]):
    #  print(src)

    return x

class Classifier(nn.Module):
  def __init__(self, d_model_tx, d_model_rw, seq_len_tx, seq_len_rw, nhead_tx, nhead_rw, dim_feedforward, nlayers_tx, nlayers_rw, device, dropout = 0.5):
    super(Classifier, self).__init__()
    self.d_model_tx = d_model_tx
    self.d_model_rw = d_model_rw
    self.seq_len_tx = seq_len_tx
    self.seq_len_rw = seq_len_rw
    self.nhead_tx = nhead_tx
    self.nhead_rw = nhead_rw
    self.dim_feedforward = dim_feedforward
    self.nlayers_tx = nlayers_tx
    self.nlayers_rw = nlayers_rw
    self.device = device
    #self.pos_encoder = PositionalEncoding(d_model, dropout)
    self.position_embedding_tx = nn.Embedding(seq_len_tx, d_model_tx)
    self.position_embedding_rw = nn.Embedding(seq_len_rw, d_model_rw)
    encoder_layer_tx = TransformerEncoderLayer(d_model_tx, nhead_tx, dim_feedforward, dropout, batch_first=True)
    encoder_layer_rw = TransformerEncoderLayer(d_model_rw, nhead_rw, dim_feedforward, dropout, batch_first=True)
    self.encoder_tx = TransformerEncoder(encoder_layer_tx, nlayers_tx)
    self.encoder_rw = TransformerEncoder(encoder_layer_rw, nlayers_rw)
    self.binary_classifier = BinaryClassification((seq_len_tx*d_model_tx + seq_len_rw*d_model_rw), device)


  def forward(self, src_tx: Tensor, src_rw: Tensor) -> Tensor:
    #print("Classifier forwrd")
    #print(src_rw)

    N, seq_length, embed_size = src_tx.shape
    positions_tx = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
    src_tx_ = src_tx + self.position_embedding_tx(positions_tx)

    N_rw, seq_length_rw, embed_size_rw = src_rw.shape
    positions_rw = torch.arange(0, seq_length_rw).expand(N_rw, seq_length_rw).to(self.device)
    src_rw_ = src_rw + self.position_embedding_rw(positions_rw)

    #print(f"src after positional embeddings: {src.shape}")
    #print(src)
    #print("before encoder")
    output_tx = self.encoder_tx(src_tx_)
    output_rw = self.encoder_rw(src_rw_)
    #print(output_rw)
    output_tx_f = torch.reshape(output_tx, (N, seq_length*embed_size))
    output_tx_rw = torch.reshape(output_rw, (N_rw, seq_length_rw*embed_size_rw))
    #print(output_tx_f)
    #print(f"encoder output shape: {output.shape}")
    #print(output)
    #print("after encoder")
    output = self.binary_classifier(torch.cat((output_tx_f, output_tx_rw), dim=1)) ##
    return output

#Loading DeFiTrust
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
#PATH = './DeFi_Latest_v4_1.pth'
#loaded_model = torch.load(PATH)
#loaded_model.eval()
#loaded_model = torch.load(PATH, map_location=device)
#loaded_model.eval()

PATH = "model.pt"
loaded_model = Classifier(d_model_tx=7, d_model_rw=5, seq_len_tx=108, seq_len_rw=100, nhead_tx=7, nhead_rw=5, dim_feedforward=16, nlayers_tx=8, nlayers_rw=8, device=device)
loaded_model.load_state_dict(torch.load(PATH, map_location=device))
loaded_model.eval()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
import numpy as np
import torch.nn as nn
bert_tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
bert_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def greet(name):
  name = name + "test"
  return "Hello " + name + "!"

def predict(token_add, Explanation):
  tx_df = eth_api.get_transfers(token_add)
  try:
    name, symbol = eth_api.get_token_name(token_add)
    print(f"{symbol} transfers collected")
    reviews = reddit_api.collect_reddit(symbol, name)
  except Exception as err:
    # some token have name and symbol as bytes32. By changing the ABI those tokens also can be collected
    print(f"Exception occured: {err}")
    print(f"abnormal address: {token_add}")
    return "abnormal token"

  feature_data = get_features(token_add, tx_df)

  print("features done")
  #print(len(feature_data))
  #print(len(feature_data[0]))

  out_l = []
  for r in reviews['review']:
    n = len(r.split())
    if n > 510:
      r = r[:510]
    try:
      tokens = bert_tokenizer.encode(r, return_tensors='pt')
      result = bert_model(tokens)
      logits = result.logits.detach().numpy()
      # print(type(logits[0]))
      lis = list(logits[0])
      # print(lis)
      out_l.append(lis)
    except:
      print("abnormal text")
  if len(out_l) < 100:
    for i in range(100 - len(out_l)):
      out_l.append([0, 0, 0, 0, 0])
  red_embed = np.array(out_l)

  review_data = []
  for i in red_embed:
    line = list(i)
    review_data.append(line)

  x1 = np.array(feature_data)
  x2 = np.array(review_data)
  np.save("demo_x1.npy", x1)
  np.save("demo_x2.npy", x2)
  x1 = np.reshape(x1, (1, 108, 7))
  x2 = np.reshape(x2, (1, 100, 5))
  x1 = torch.from_numpy(x1)
  x2 = torch.from_numpy(x2)

  sigmoid = nn.Sigmoid()
  model_out = loaded_model(x1.float(), x2.float())
  val = sigmoid(model_out)
  prediction = val.detach().numpy()[0][0]
  print(prediction)

  if (Explanation):
    ig = initiate_ig(loaded_model)
    eth_attr, sent_attr = generate_ig(x1, x2, ig)
    gradient_all(eth_attr)
    im_grad_all = Image.open("gradient_all.png")
    value_gradient(eth_attr, x1.cpu().numpy().reshape(108,7), 'Block_Dif')
    im_grad_val = Image.open("val_grad.png")
    scatter_sentiment_grad(sent_attr, x2.cpu().numpy().reshape(100,5))
    im_sent_val = Image.open("sentiment_vals.png")
    im_sent_grad = Image.open("sentiment_grad.png")
  else:
    im_grad_all = np.zeros((1429, 4839))
    im_grad_val = np.zeros((1349, 5056))
    im_sent_val = np.zeros((1349, 5056))
    im_sent_grad = np.zeros((1349, 5056))

  if prediction<0.5:
    return "Scam: " + str(prediction), im_grad_all, im_grad_val, im_sent_val, im_sent_grad
  else:
    return "Trustworthy: " + str(prediction), im_grad_all, im_grad_val, im_sent_val, im_sent_grad

  #return str(torch.round(val))
  #return "fun"


demo = gr.Interface(fn=predict, inputs=["text", "checkbox"], outputs=["text","image","image","image","image"])

demo.launch()

# #testing BERT(Delete this)
# df1 = pd.read_csv('ankh_reviews.csv')
#
# out_l = []
# for r in df1['review']:
#   n = len(r.split())
#   if n>510:
#     r = r[:510]
#   try:
#     tokens = bert_tokenizer.encode(r, return_tensors='pt')
#     result = bert_model(tokens)
#     logits = result.logits.detach().numpy()
#     #print(type(logits[0]))
#     lis = list(logits[0])
#     #print(lis)
#     out_l.append(lis)
#   except:
#     print("abnormal text")
# if len(out_l)<100:
#   for i in range(100 - len(out_l)):
#     out_l.append([0,0,0,0,0])
# out_ar = np.array(out_l)
#
# print(out_ar)

# #testing DeFiTrust (Delete this)
# sushi_x1 = np.load("sushi_full_x1.npy", allow_pickle=True)
# sushi_x2 = np.load("sushi_full_x2.npy", allow_pickle=True)
#
# sushi_x1 = np.reshape(sushi_x1, (1, 108, 7))
# sushi_x2 = np.reshape(sushi_x2, (1, 100, 5))
# sushi_x1 = torch.from_numpy(sushi_x1)
# #sushi_x1 = sushi_x1.to(device)
# sushi_x2 = torch.from_numpy(sushi_x2)
# #sushi_x2 = sushi_x2.to(device)
#
# sigmoid = nn.Sigmoid()
# model_out = loaded_model(sushi_x1.float(), sushi_x2.float())
# val = sigmoid(model_out)
# print(val)
# print(torch.round(val))

#testing eth_collect
# token_add = "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"
# tx_df = eth_api.get_transfers(token_add)
# tx_df.to_csv("testingdf.csv", index=False)

