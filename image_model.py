"""HW04.ipynb

Original file is located at
    https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW04/HW04.ipynb
  - Hard: Construct [conformer](https://arxiv.org/abs/2005.08100) which is a variety of transformer. 
"""

import os
import time
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from resnet import resnet152
from conformer import ConformerConvModule

timestr = time.strftime("%Y_%m%d_%H%M")
folder_path = "./model/image_%s" % timestr

print(folder_path)

try:
    os.mkdir(folder_path)
except OSError:
    print ("Creation of the directory %s failed" % folder_path)
else:
    print ("Successfully created the directory %s " % folder_path)

with open(__file__, 'r') as f:
    with open("%s/image_model.py" % folder_path, 'w') as out:
        out.write(f.read())

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from image_dataset import FaceTouchImageDataset
import utils

def get_dataloader(idx, batch_size, n_workers):
  """Generate dataloader"""
  train_dict = utils.get_dataset(idx, train=True)
  # test_dict = utils.get_dataset(idx, train=False)
  dataset = FaceTouchImageDataset(train_dict)
  trainlen = int(0.9 * len(dataset))
  lengths = [trainlen, len(dataset) - trainlen]
  trainset, validset = random_split(dataset, lengths)
  
  # validset = FaceTouchDataset(test_dict)

  train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers,
    pin_memory=True
  )
  valid_loader = DataLoader(
    validset,
    batch_size=batch_size,
    num_workers=n_workers,
    drop_last=True,
    pin_memory=True,
    shuffle=True
  )

  return train_loader, valid_loader

import torch
import torch.nn as nn
import torch.nn.functional as F

class Image_Classifier(nn.Module):
  def __init__(self, d_model=128, dropout=0.1):
    super().__init__()
    # Resnet
    self.resnet = resnet152(pretrained=True)
    # Project the dimension of features from that of input into d_model.
    self.L1 = nn.Linear(2048, d_model)
    self.L2 = nn.Linear(d_model, int(d_model/4))
    self.L3 = nn.Linear(int(d_model/4), 2)
    self.Relu = nn.ReLU()


  def forward(self, mels):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    out = mels.view(-1, 3, 224, 224)
    out = self.resnet(out)
    out = self.L1(out)
    out = self.Relu(out)
    out = self.L2(out)
    out = self.Relu(out)
    out = self.L3(out)
    return out

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch


def model_fn(batch, model, criterion, device):
  """Forward a batch through the model."""

  mels, labels = batch
  mels = mels.to(device)
  labels = labels.to(device)

  outs = model(mels)

  loss = criterion(outs, labels)

  # Get the speaker id with highest probability.
  preds = outs.argmax(1)
  # Compute accuracy.
  accuracy = torch.mean((preds == labels).float())

  return loss, accuracy

from tqdm import tqdm
import torch


def valid(dataloader, model, criterion, device): 
  """Validate on validation set."""

  model.eval()
  running_loss = 0.0
  running_accuracy = 0.0
  pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

  for i, batch in enumerate(dataloader):
    with torch.no_grad():
      loss, accuracy = model_fn(batch, model, criterion, device)
      running_loss += loss.item()
      running_accuracy += accuracy.item()

    pbar.update(dataloader.batch_size)
    pbar.set_postfix(
      loss=f"{running_loss / (i+1):.2f}",
      accuracy=f"{running_accuracy / (i+1):.2f}",
    )

  pbar.close()
  model.train()

  return running_accuracy / len(dataloader)

"""# Main function"""

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split


def parse_args():
  """arguments"""
  config = {
    "index": 0,
    "save_path": '%s/model.ckpt' % folder_path,
    "batch_size": 32,
    "n_workers": 8,
    "valid_steps": 1000,
    "warmup_steps": 1000,
    "save_steps": 1000,
    "total_steps": 40000,
  }

  return config

def main(
  index,
  save_path,
  batch_size,
  n_workers,
  valid_steps,
  warmup_steps,
  total_steps,
  save_steps,
):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  train_loader, valid_loader = get_dataloader(index, batch_size, n_workers)
  train_iterator = iter(train_loader)
  print(f"[Info]: Finish loading data!",flush = True)

  model = Image_Classifier().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=1e-3)
  # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
  print(f"[Info]: Finish creating model!",flush = True)

  best_accuracy = -1.0
  best_state_dict = None

  pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

  for step in range(total_steps):
    # Get data
    try:
      batch = next(train_iterator)
    except StopIteration:
      train_iterator = iter(train_loader)
      batch = next(train_iterator)

    loss, accuracy = model_fn(batch, model, criterion, device)
    batch_loss = loss.item()
    batch_accuracy = accuracy.item()

    # Updata model
    loss.backward()
    optimizer.step()
    # scheduler.step()
    optimizer.zero_grad()
    
    # Log
    pbar.update()
    pbar.set_postfix(
      loss=f"{batch_loss:.2f}",
      accuracy=f"{batch_accuracy:.2f}",
      step=step + 1,
    )

    # Do validation
    if (step + 1) % valid_steps == 0:
      pbar.close()

      valid_accuracy = valid(valid_loader, model, criterion, device)

      # keep the best model
      if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        best_state_dict = model.state_dict()
        with open("%s/log.txt" % folder_path, 'a') as f:
          f.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.5f})\n")
        
      pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    # Save the best model so far.
    if (step + 1) % save_steps == 0 and best_state_dict is not None:
      torch.save(best_state_dict, save_path)
      pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

  pbar.close()


if __name__ == "__main__":
  main(**parse_args())


import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

import json
import csv
from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
import torch
from torch.utils.data import DataLoader

def parse_args():
  """arguments"""
  config = {
    "idx": 0,
    "model_path": '%s/model.ckpt' % folder_path,
    "output_path": "%s/output.csv" % folder_path,
  }

  return config


def main(
  idx,
  model_path,
  output_path,
):
  """Main function."""
  criterion = nn.CrossEntropyLoss()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  test_dict = utils.get_dataset(idx, train=False)
  dataset = FaceTouchImageDataset(test_dict)
  dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=8,
  )
  print(f"[Info]: Finish loading data!",flush = True)

  model = Image_Classifier().to(device)
  model.load_state_dict(torch.load(model_path))
  model.eval()
  print(f"[Info]: Finish creating model!",flush = True)
  
  preds_all = []
  labels_all = []
  for batch in tqdm(dataloader):
    with torch.no_grad():
      mels = batch[0]
      labels = batch[1]
      mels = mels.to(device)
      outs = model(mels)
      preds = outs.argmax(1).cpu().numpy()
      preds_all.extend(list(preds))
      labels_all.extend(list(labels))

  accuracy = accuracy_score(labels_all, preds_all)
  f_score = f1_score(labels_all, preds_all)
  mcc = matthews_corrcoef(labels_all, preds_all)
  print("acc: ", accuracy)
  print("f_score: ", f_score)
  print("mcc: ", mcc)

  with open("%s/log.txt" % folder_path, 'a') as f:
    f.write(f"(accuracy={accuracy:.5f})\n")
    f.write(f"(f score={f_score:.5f})\n")
    f.write(f"(mcc={mcc:.5f})\n")
  
if __name__ == "__main__":
  main(**parse_args())