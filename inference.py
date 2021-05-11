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
from img_based import Classifier2
from dataset import FaceTouchDataset

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from transformer import Classifier

# folder_path = "model/transformer_2021_0427_0303"
# folder_path = "model/transformer_2021_0427_0437"
# folder_path = "model/transformer_2021_0427_0603"
folder_path = "model/transformer_2021_0427_1059"
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