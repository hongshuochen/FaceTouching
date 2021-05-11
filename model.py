import os
import cv2
import time
import json
import pickle
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt

from collections import OrderedDict

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def model(vid_id, draw=False):
    json_path = "../face_touch_annotations/Pose estimations/json"
    vid = os.listdir(json_path)[vid_id]
    print(vid)
    landmark_path = "landmarks" + '/' + vid.split('.')[0] + '/' + vid.split('.')[0] + '.csv'
    df = pd.read_csv(landmark_path)
    x = np.array(df.iloc[:,5:5+68])
    y = np.array(df.iloc[:,5+68:5+68*2])
    conf = np.array(df.iloc[:,3])
    x_ = x[...,np.newaxis]
    y_ = y[...,np.newaxis]
    landmarks = np.concatenate([x_,y_],axis=-1)

    with open(os.path.join(json_path, vid)) as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    labels = []
    preds = []
    record_idx = []
    missed = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

    for idx in range(len(x)-2):
        if len(data[idx+2]['faces']) == 0:
            continue
        label = data[idx+2]['faces'][0]['frame_label']
        # Capture frame-by-frame
        frame = cv2.imread("frames/" + vid.replace('.json', '') + '/' + str(idx+1).zfill(5) + ".png")
        points1 = np.array(data[idx+2]['right_hands'])[:,:,:2]/2
        points2 = np.array(data[idx+2]['left_hands'])[:,:,:2]/2
        face_x = x[idx]
        face_y = y[idx]
        face_x_contour = np.concatenate([face_x[:17], face_x[17:27][::-1]])
        face_y_contour = np.concatenate([face_y[:17], face_y[17:27][::-1]])
        
        landmark = landmarks[idx]
        leftEyePts = landmark[lStart:lEnd]
        rightEyePts = landmark[rStart:rEnd]
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        normalized_dist = LA.norm(leftEyeCenter - rightEyeCenter)
        face_center = np.mean(landmark, axis=0).astype("int")

        contour = []
        for i in range(len(face_x_contour)):
            contour.append([[face_x_contour[i],face_y_contour[i]]])
        contours = [np.array(contour).astype(int)]
        if draw:
            frame = cv2.drawContours(frame.copy(), contours, 0, (0,0,255), 3)
            for j in range(len(face_x)):
                image = cv2.circle(frame, (int(face_x[j]), int(face_y[j])), 3, (0, 0, 255), -1)
        pred = False
        warning = False
        dists = []
        for i in range(len(points1)):
            for center_coordinates in points1[i]:
                if draw:
                    image = cv2.circle(frame, tuple(center_coordinates.astype(int)), 3, (255, 0, 0), -1)
                dist = cv2.pointPolygonTest(contours[0], tuple(center_coordinates.astype(int)), True)
                dists.append(dist)
                if dist >= 0:
                    pred = True
                if dist >= -normalized_dist*5:
                    warning = True
                # if not warning:
                #     for p in landmark:
                #         dist_face = LA.norm(p-center_coordinates.astype(int))
                #         if dist_face < 100:
                #             warning = True
        if draw:
            image = cv2.putText(image, str(round(sum(dists)/len(dists),2)), (300,300), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        dists = []
        for i in range(len(points2)):
            for center_coordinates in points2[i]:
                if draw:
                    image = cv2.circle(frame, tuple(center_coordinates.astype(int)), 3, (0, 255, 0), -1)
                dist = cv2.pointPolygonTest(contours[0], tuple(center_coordinates.astype(int)), True)
                dists.append(dist)
                if dist >= 0:
                    pred = True
                if dist >= -normalized_dist*5:
                    warning = True
                # if not warning:
                #     for p in landmark:
                #         dist_face = LA.norm(p-center_coordinates.astype(int))
                #         if dist_face < 100:
                #             warning = True
        if draw:
            image = cv2.putText(image, str(round(sum(dists)/len(dists),2)), (300,350), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if draw:
            if pred == True:
                text = 'Touch'
                image = cv2.putText(image, text, (100,100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                text = 'Non-Touch'
                image = cv2.putText(image, text, (100,100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        labels.append(label)
        preds.append(pred)
        if pred:
            warning = 1
        if label == 1 and warning == 0:
            print("missed")
            missed.append(idx+2)
        print(idx, end='\r')
        # Display the resulting frame
        if draw:
            cv2.imshow('image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if warning:
            record_idx.append(idx+2)
    if draw:
        cv2.destroyAllWindows()
    pickle.dump(record_idx, open("warning/" + vid.replace('.json', '.pkl'),"wb"))
    pickle.dump(missed, open("missed/" + vid.replace('.json', '.pkl'),"wb"))
    return labels, preds

if __name__ == '__main__':
    import multiprocessing
    params = []
    for vid_id in range(64):
        if vid_id not in [17, 20, 21, 24, 28, 31, 34, 36, 48, 50, 62]:
            params.append([vid_id])
    pool = multiprocessing.Pool(8)
    pool.starmap(model, params)
    # label, pred = model(vid_id)
