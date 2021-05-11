#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
	IIT : Istituto italiano di tecnologia
    Pattern Analysis and Computer Vision (PAVIS) research line
    Usage Example:
		$ python3
    LICENSE:
	This project is licensed under the terms of the MIT license.
	This project incorporates material from the projects listed below (collectively, "Third Party Code").
	This Third Party Code is licensed to you under their original license terms.
	We reserves all other rights not expressly granted, whether by implication, estoppel or otherwise.
	The software can be freely used for any non-commercial applications.
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

import torch
from torch.utils.data import Dataset
from facealigner import FaceAligner
from torchvision import transforms

# Defining the pyTorch custom dataset
class FaceTouchImageDataset(Dataset):
    def __init__(self, dictionary, transform=None, log_enabled=False):
        train_dict = {
            "imgs": dictionary['imgs'],
            "labels": dictionary['labels']
        }
        self.img_filenames = dictionary['imgs']
        self.labels = dictionary['labels']
        self.transform = transform

        self.log_enabled = log_enabled
        self.preprocess = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.df_dict = {}
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.img_filenames[idx]
        frame_num = filename.split('/')[-1].split('.')[0]

        framename = filename.replace(frame_num, str(int(frame_num)).zfill(5))
        vid = framename.split('/')[-2]
        landmark_path = "landmarks" + '/' + vid + '/' + vid + '.csv'
        if vid not in self.df_dict:
            if os.path.isfile(landmark_path):
                df = pd.read_csv(landmark_path)
                self.df_dict[vid] = df
        if os.path.isfile(framename) and os.path.isfile(landmark_path):
            img = Image.open(framename)
            if not img.mode == 'RGB':
                img = img.convert('RGB')
            img = np.array(img)
            img = self.align(img, framename, int(frame_num))

        else:
            img = Image.fromarray(np.zeros([224, 224, 3], dtype=np.uint8))
            img = np.array(img)
        img = torch.from_numpy(img)
        img = img.float()
        img /= 255
        img = img.permute(2, 0, 1)
        img = self.preprocess(img)
        if self.transform:
            img = self.transform(img)
        img = np.array(img)

        label = int(self.labels[idx])
        label = np.array(label)
        
        return (img, label)

    def __len__(self):
        return len(self.img_filenames)
    
    def align(self, img, framename, frame_num):
        vid = framename.split('/')[-2]
        df = self.df_dict[vid]
        x = np.array(df.iloc[frame_num-1,5:5+68]).reshape(68,-1)
        y = np.array(df.iloc[frame_num-1,5+68:5+68*2]).reshape(68,-1)
        landmarks = np.concatenate((x,y), axis=1)
        aligner = FaceAligner(desiredLeftEye=(0.42, 0.42), desiredFaceWidth=224)
        aligned_face = aligner.align(img, landmarks)
        return aligned_face