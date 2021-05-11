import os
import numpy as np
import multiprocessing

def landmark_extraction(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Input: ", image_dir)
    cmd = "./OpenFace/build/bin/FeatureExtraction -fdir '{input}' -out_dir '{output}' -2Dfp -aus"\
        .format(input = image_dir, output = output_dir)
    print(cmd)
    stream = os.popen(cmd)
    output = stream.read()
    print("Output:", output_dir)
    return True

if __name__ == '__main__':
    path = "frames"
    params = []
    for vid in os.listdir(path):
        params.append([os.path.join(path, vid), os.path.join("landmarks", vid)])
    pool = multiprocessing.Pool(8)
    pool.starmap(landmark_extraction, params)