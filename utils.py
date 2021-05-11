import os
import json
import pickle
import numpy as np
from tqdm import tqdm

def get_fold(idx):
    folds_path = "folds"
    fold_list = []
    for fold in os.listdir(folds_path):
        fold_list.append([])
        for vid in os.listdir(os.path.join(folds_path, fold)):
            fold_list[-1].append(vid.split('.')[0].replace('-','_'))
    return fold_list[idx]

def get_dataset(idx, train=True):
    fold = get_fold(idx)
    # print(fold)
    # json_path = "../face_touch_annotations/Pose estimations/json"
    f = os.listdir("warning")
    labels = []
    imgs = []
    for file in f:
        if train:
            if file.split('.')[0] in fold:
                continue
        else:
            if file.split('.')[0] not in fold:
                continue
        # print(file.split('.')[0])
        warning = pickle.load(open("warning/" + file, "rb"))
    
        # with open(os.path.join(json_path, file.replace('pkl','json'))) as j:
        #     data = []
        #     for line in j:
        #         data.append(json.loads(line))
        data = pickle.load(open("labels/" + file,"rb"))

        for i in warning:
            label = data[i]
            labels.append(label)
            imgs.append("frames/" + file.replace('.pkl', '') + '/' + str(i-1).zfill(5) + ".png")
    
    imgs = np.array(imgs)
    labels = np.array(labels)

    if train:
        imgs, labels = get_balanced_datset(imgs, labels)

    dictionary = dict()
    dictionary['imgs'] = imgs
    dictionary['labels'] = labels
    print(len(imgs))
    return dictionary

def get_balanced_datset(imgs, labels):
    if sum(labels) < len(labels) - sum(labels):
        touch_imgs = imgs[labels == 1]
        no_touch_imgs = imgs[labels == 0]
        idx = np.random.choice(len(no_touch_imgs), len(touch_imgs), replace=False)
        no_touch_imgs = no_touch_imgs[idx]
        labels = [1]*len(touch_imgs)
        labels.extend([0]*len(no_touch_imgs))
        imgs = np.concatenate([touch_imgs, no_touch_imgs], axis=0)
        labels = np.array(labels)
    return imgs, labels

# json_path = "../face_touch_annotations/Pose estimations/json"
# f = os.listdir("warning")
# for file in f:
#     print(file)
#     with open(os.path.join(json_path, file.replace('pkl','json'))) as j:
#         data = []
#         for line in j:
#             data.append(json.loads(line))
#         labels = []
#         for x in data:
#             if len(x['faces']) == 1:
#                 label = x['faces'][0]['frame_label']
#             else:
#                 label = 0
#             labels.append(label)
#     pickle.dump(labels, open("labels/" + file,"wb"))

# json_path = "../face_touch_annotations/Pose estimations/json"
# f = os.listdir("warning")
# for file in f:
#     print(file)
#     with open(os.path.join(json_path, file.replace('pkl','json'))) as j:
#         data = []
#         for line in j:
#             data.append(json.loads(line))
#     pickle.dump(data, open("keypoints/" + file,"wb"))