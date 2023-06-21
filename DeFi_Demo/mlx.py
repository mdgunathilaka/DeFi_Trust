from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
import numpy as np

def initiate_ig(model):
    return IntegratedGradients(model)

def generate_ig(sample_1, sample_2, ig):
  baseline1 = torch.zeros(sample_1.shape[0], sample_1.shape[1], sample_1.shape[2])
  baseline2 = torch.zeros(sample_2.shape[0], sample_2.shape[1], sample_2.shape[2])
  attributions, approximation_error = ig.attribute((sample_1.float(), sample_2.float()),
                                                 baselines=(baseline1, baseline2),
                                                 method='gausslegendre',
                                                 return_convergence_delta=True)
  return attributions[0].cpu().numpy().reshape(108,7), attributions[1].cpu().numpy().reshape(100,5)

def gradient_all(attr):
  max_int = abs(max(list(np.reshape(attr, (108*7))), key=abs))
  fig = plt.figure(figsize = (20, 5))
  x = np.array([i for i in range(1,109)])
  plt.xticks([i for i in range(1,109,2)])
  features = ['Block_Dif', 'Volume', 'Minted', 'Burnt', 'Unique_add', 'Gene_Coef', 'Avg_Gas']

  for i in range(7):
    col = []
    for j in range(108):
      col.append(abs(attr[j][i]))
    #col = list(map(lambda x: x/max_int, col))
    #col = list(map(lambda x_: x_/abs(max(col, key=abs)), col))
    plt.xticks([i for i in range(1,109,2)])
    plt.plot(x, col)

  plt.xlabel("Timesteps", fontsize=14)
  plt.ylabel("Value", fontsize=14)
  plt.title("Gradients of all features", fontsize=14)
  plt.legend(features, fontsize=14)
  plt.savefig('gradient_all.png', dpi=300, bbox_inches='tight')
  #plt.savefig('gradient_all_trust.pdf', dpi=1200, bbox_inches='tight')
  #plt.show()
  plt.close()

def value_gradient(attr, vals, feature):
  ##
  #pdf = PdfPages('line_plot.pdf')
  features = ['Block_Dif', 'Volume', 'Minted', 'Burnt', 'Unique_add', 'Gene_Coef', 'Avg_Gas']
  f_index = features.index(feature)
  steps = [i for i in range(1,109)]
  fig = plt.figure(figsize = (20, 5))
  values = []
  for arr in vals:
    values.append(arr[f_index])
  grad_vals = []
  for arr in attr:
    grad_vals.append(arr[f_index])

  ax1 = plt.subplot()
  l1, = ax1.plot(steps, values, color='red')
  ax2 = ax1.twinx()
  l2, = ax2.plot(steps, grad_vals, color='blue')
  ax1.set_ylabel("Value", fontsize=14)
  ax2.set_ylabel("IG Value", fontsize=14)
  plt.legend([l1, l2], ["Feature_Value", "IG_Value"], fontsize=14)
  plt.title("Importance of "+feature)
  plt.xticks([i for i in range(1,109,2)])
  ##
  plt.savefig('val_grad.png', dpi=300, bbox_inches='tight')
  #pdf.close()
  #plt.show()
  plt.close()

def scatter_sentiment_grad(attr, vals):
  #pdf1 = PdfPages('sentiment_v.pdf')
  #pdf2 = PdfPages('sentiment_g.pdf')
  fig = plt.figure(figsize = (20, 5))
  plt.xticks([i for i in range(1,101,2)])
  x = np.array([i for i in range(1,101)])
  values_i = []
  values = []
  grad_i = []
  grad_vals = []
  for i in range(attr.shape[0]):
    max_v = vals[i].argmax()
    values_i.append(max_v/4)
    values.append(vals[i][max_v])
    max_g = np.abs(attr[i]).argmax()
    grad_i.append(max_g/4)
    grad_vals.append(attr[i][max_g])
  values = list(map(lambda x_: x_/abs(max(values, key=abs)), values))
  grad_vals = list(map(lambda x_: x_/abs(max(grad_vals, key=abs)), grad_vals))
  plt.scatter(x, values, s=75, c=values_i, cmap='Reds', linewidths=0.4, edgecolors='black')
  #plt.colorbar()
  plt.xlabel("Timesteps", fontsize=14)
  plt.ylabel("Value", fontsize=14)
  plt.title("Values of Sentiment", fontsize=14)
  plt.legend(["Value"], fontsize=14)
  #pdf1.savefig(fig, dpi=1200)
  plt.savefig('sentiment_vals.png', dpi=300, bbox_inches='tight')
  plt.close()

  fig = plt.figure(figsize = (20, 5))
  plt.xticks([i for i in range(1,101,2)])
  plt.scatter(x, grad_vals, s=75, c=grad_i, cmap='Blues', linewidths=0.4, edgecolors='black')
  #plt.colorbar()
  plt.xlabel("Timesteps", fontsize=14)
  plt.ylabel("Sentiment", fontsize=14)
  plt.title("Importance of Sentiment", fontsize=14)
  plt.legend(["IG Value"], fontsize=14)
  #pdf2.savefig(fig, dpi=1200)
  #pdf1.close()
  #pdf2.close()
  plt.savefig('sentiment_grad.png', dpi=300, bbox_inches='tight')
  plt.close()
