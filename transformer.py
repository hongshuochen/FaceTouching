"""Transformer

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
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet152
from conformer import ConformerConvModule

import utils
from image_model import Image_Classifier
from dataset import FaceTouchDataset


timestr = time.strftime("%Y_%m%d_%H%M")
folder_path = "./model/transformer_%s" % timestr

try:
    os.mkdir(folder_path)
except OSError:
    print ("Creation of the directory %s failed" % folder_path)
else:
    print ("Successfully created the directory %s " % folder_path)

with open(__file__, 'r') as f:
    with open("%s/transformer.py" % folder_path, 'w') as out:
        out.write(f.read())

def get_dataloader(idx, batch_size, n_workers):
  """Generate dataloader"""
  train_dict = utils.get_dataset(idx, train=True)
  dataset = FaceTouchDataset(train_dict)
  trainlen = int(0.95 * len(dataset))
  lengths = [trainlen, len(dataset) - trainlen]
  trainset, validset = random_split(dataset, lengths)

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
    shuffle=True,
    num_workers=n_workers,
    drop_last=True,
    pin_memory=True,
  )

  return train_loader, valid_loader

class Classifier(nn.Module):
  def __init__(self, d_model=80, dropout=0.1):
    super().__init__()
    # Resnet
    model_path = "model/image_2021_0509_0922/model.ckpt"
    model = Image_Classifier()
    model.load_state_dict(torch.load(model_path))
    for param in model.parameters():
      param.requires_grad = False
    self.resnet = model
    # Project the dimension of features from that of input into d_model.
    self.prenet = nn.Linear(2, d_model)
    # TODO:
    #   Change Transformer to Conformer.
    #   https://arxiv.org/abs/2005.08100
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=256, nhead=1, 
    )
    self.conformer = ConformerConvModule(
      dim = d_model,
      causal = False,             # auto-regressive or not - 1d conv will be made causal with padding if so
      expansion_factor = 2,       # what multiple of the dimension to expand for the depthwise convolution
      kernel_size = 3,           # kernel size, 17 - 31 was said to be optimal
      dropout = dropout                # dropout at the very end
    )

    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    # Project the the dimension of features from d_model into two classes.
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.ReLU(),
      nn.Linear(d_model, 2),
    )

  def forward(self, mels):
    """
    args:
      mels: (batch size, 5, 3, 224, 224)
    return:
      out: (batch size, 2)
    """
    out = mels.view(-1, 3, 224, 224)
    out = self.resnet(out)
    out = out.view(-1, 11, 2)
    # out: (batch size, length, d_model)
    out = self.prenet(out)
    # out: (length, batch size, d_model)
    out = out.permute(1, 0, 2)
    # The encoder layer expect features in the shape of (length, batch size, d_model).
    out = self.encoder_layer(out)
    # out = self.encoder(out)
    # out = self.conformer(out)
    
    # out: (batch size, length, d_model)
    out = out.transpose(0, 1)
    # mean pooling
    stats = out.mean(dim=1)

    # out: (batch, 2)
    out = self.pred_layer(stats)
    return out

"""# Learning rate schedule
- For transformer architecture, the design of learning rate schedule is different from that of CNN.
- Previous works show that the warmup of learning rate is useful for training models with transformer architectures.
- The warmup schedule
  - Set learning rate to 0 in the beginning.
  - The learning rate increases linearly from 0 to initial learning rate during warmup period.
"""

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
  optimizer: Optimizer,
  num_warmup_steps: int,
  num_training_steps: int,
  num_cycles: float = 0.5,
  last_epoch: int = -1,
):
  """
  Create a schedule with a learning rate that decreases following the values of the cosine function between the
  initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
  initial lr set in the optimizer.

  Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

  Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
  """

  def lr_lambda(current_step):
    # Warmup
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    # decadence
    progress = float(current_step - num_warmup_steps) / float(
      max(1, num_training_steps - num_warmup_steps)
    )
    return max(
      0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

  return LambdaLR(optimizer, lr_lambda, last_epoch)

"""# Model Function
- Model forward function.
"""

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

"""# Validate
- Calculate accuracy of the validation set.
"""

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
    "valid_steps": 2000,
    "warmup_steps": 1000,
    "save_steps": 2000,
    "total_steps": 10000,
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

  model = Classifier().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=1e-3)
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
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
    scheduler.step()
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
import time
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet152
from conformer import ConformerConvModule
from tqdm import tqdm
import utils
from image_model import Image_Classifier
from dataset import FaceTouchDataset

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from transformer import Classifier

def parse_args():
  """arguments"""
  config = {
    "idx": 0,
    "model_path": '%s/model.ckpt' % folder_path,
  }

  return config


def main(
  idx,
  model_path
):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  test_dict = utils.get_dataset(idx, train=False)
  dataset = FaceTouchDataset(test_dict)
  dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=8,
  )
  print(f"[Info]: Finish loading data!",flush = True)

  model = Classifier().to(device)
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