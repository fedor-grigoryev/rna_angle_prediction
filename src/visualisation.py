from os import listdir
import pandas as pd

import matplotlib.pyplot as plt


def visualise_distr(train_data_path):
    RNA_LIST = {i: name for i, name in enumerate(
        listdir(train_data_path))}
    gamma = pd.read_csv(f"{train_data_path}/1CSL.csv")["gamma"]
    plt.hist(gamma, label="1CSL gamma distribution")
    plt.xlabel("Degrees")
    plt.ylabel("Number of Angles")
    plt.show()
