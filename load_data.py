import os
from dataset import FaceTouchDataset

path = "frames"
videos = os.listdir(path)

imgs = []
labels = []
count = 0
for vid in videos:
    vid_path = os.path.join(path, vid)
    frames = os.listdir(vid_path)
    for i in frames:
        image_path = os.path.join(vid_path, i)
        print(image_path)
        imgs.append(image_path)
        labels.append(0)
        count += 1
        if count%1000 == 0:
            break
    if count == 10000:
        break

dictionary = dict()
dictionary['imgs'] = imgs
dictionary['labels'] = labels

data = FaceTouchDataset(dictionary)
ims, _ = data.__getitem__(5000)
from matplotlib import pyplot as plt

plt.imshow(ims[0][0])
plt.show()
plt.imshow(ims[1][0])
plt.show()
plt.imshow(ims[2][0])
plt.show()
plt.imshow(ims[3][0])
plt.show()
plt.imshow(ims[4][0])
plt.show()
print(len(ims))