import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

import json
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import utils
from img_based import Classifier
from dataset2 import FaceTouchDataset
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
  dataset = FaceTouchDataset(test_dict)
  dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
  )
  print(f"[Info]: Finish loading data!",flush = True)

  model = Classifier().to(device)
  model.load_state_dict(torch.load(model_path))
  model.eval()
  print(f"[Info]: Finish creating model!",flush = True)
  

folder_path = "model/2021_0426_2005"
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
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  test_dict = utils.get_dataset(idx, train=True)
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

  for img, label, filename in tqdm(dataloader):
    with torch.no_grad():
      img = img.to(device)
      outs = model(img)
      outs = outs.cpu().numpy()
      for f in enumerate(filename):
        if not os.path.exists(os.path.dirname(str(filename[0])).replace("frames", "features")):
            os.makedirs(os.path.dirname(str(filename[0])).replace("frames", "features"))
        pickle.dump(outs, open(filename[0].replace("frames", "features").replace("png","pkl"), "wb"))
      
if __name__ == "__main__":
  main(**parse_args())