import os
import cv2
import numpy as np
import multiprocessing

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Input: ", video_path)
    print("ffmpeg -i '{input}' -vf scale=640:512  -r 20 '{output}/%05d.png'".format(input = video_path, output = output_dir))
    stream = os.popen("ffmpeg -i '{input}' -vf scale=640:512  -r 20 '{output}/%05d.png'" 
                      .format(input = video_path, output = output_dir))
    output = stream.read()
    print("Output:", output_dir)
    return True

if __name__ == '__main__':
    path = "../LeadershipCorpus_PAVIS_IIT/Videos_with_Audio"
    json_path = "../face_touch_annotations/Pose estimations/json"
    params = []
    print(len(os.listdir(json_path)))
    for vid in os.listdir(json_path):
        print(vid)
        keywords = vid.replace('.json', '').split('_')
        vid_path = os.path.join(path, keywords[0], keywords[1])
        for x in os.listdir(vid_path):
            if keywords[-1] in x.split(' ')[3]:
                vid_path = os.path.join(vid_path, x)
                print(vid_path)
                vid = vid.replace('.json','')
                params.append([vid_path, os.path.join("frames", vid)])
                break
    pool = multiprocessing.Pool(8)
    pool.starmap(extract_frames, params)