import os
from scipy.io import loadmat, savemat
import numpy as np
from numpy import copy

def main():
    d = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 7:6, 8:7, 9:8, 10:9}
    features_dir = "features_dir/"
    for mat_file in os.listdir(features_dir):
        print(mat_file)
        t = loadmat(features_dir + mat_file)
        Y = t["Y"][:, 1]
        print(np.unique(Y))
        y_tmp = copy(Y)
        for k, v in d.items(): y_tmp[Y == k] = v
        t["Y"] = y_tmp.reshape(-1,1)
        savemat(features_dir + mat_file, t)

if __name__ == "__main__":
    main()