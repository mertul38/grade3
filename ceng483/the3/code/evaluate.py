import numpy as np
from utils import read_image
import sys
from matplotlib import pyplot as plt


def main(argv):
    mean_squared_error(argv)

def mean_squared_error(argv):
    if len(argv) != 2:
        print("Usage: python evaluate.py estimations.npy img_names.txt")
        exit()

    estimations = np.load(argv[0])
    ground_truths = np.load(argv[1])

    acc = 0
    for est, truth in zip(estimations, ground_truths):
        cur = truth.reshape(-1).astype(np.int64)
        est = est.reshape(-1).astype(np.int64)

        cur_acc = ((cur - est) ** 2).sum() / cur.shape[0]
        acc += cur_acc
    acc /= len(estimations)
    print(f"{acc:.2f}/1.00")

def loss_12(argv):
    if len(argv) != 2:
        print("Usage: python evaluate.py estimations.npy img_names.txt")
        exit()

    estimations = np.load(argv[0])
    ground_truths = np.load(argv[1])

    acc = 0
    for est, truth in zip(estimations, ground_truths):
        cur = truth.reshape(-1).astype(np.int64)
        est = est.reshape(-1).astype(np.int64)

        cur_acc = (np.abs(cur - est) < 12).sum() / cur.shape[0]
        acc += cur_acc
    acc /= len(estimations)
    print(f"{acc:.2f}/1.00")

if __name__ == "__main__":
    main(sys.argv[1:])
