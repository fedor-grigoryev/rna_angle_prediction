from os import listdir
import pandas as pd

import matplotlib.pyplot as plt


def visualise_distr(train_data_path):
    RNA_LIST = {i: name for i, name in enumerate(
        listdir(train_data_path))}
    gammas=[]
    for elem in RNA_LIST_test:
      gammas.append(np.array(pd.read_csv(train_data_path+RNA_LIST[elem])["gamma"]))
    for i in range(len(gammas_test)):
      plt.hist(gammas[i])
    plt.xlabel("Degrees")
    plt.ylabel("Number of Angles")
    #plt.show()


def visualise_len_distr(train_data_path):
    RNA_LIST = {i: name for i, name in enumerate(
        listdir(train_data_path))}
    sequences=[]
    for elem in RNA_LIST_test:
      sequences.append(np.array(pd.read_csv(folder_test+RNA[elem])["base"].str[-1]))
    seq_ln=[]
    for i in range(len(sequences)):
      seq=''
      for elem in sequences[i]:
        seq+=elem.upper()
      seq_ln.append(len(seq))
    plt.hist(seq_ln)
    plt.xlabel("Length")
    plt.ylabel("Number of Sequences")
