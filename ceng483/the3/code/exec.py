import matplotlib.pyplot as plt

from train_and_test import train

RUN_TRAIN = True
CREATE_VAL = False
CREATE_TEST = False


if __name__ == "__main__":
    if RUN_TRAIN:
        train()
