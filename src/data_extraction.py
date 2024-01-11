from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np

import json

# path_train = "../data/angles/TrainingSet", path_test = "../data/angles/TestSet"


def train_test_extraction(train_data_path, test_data_path):
    gammas_train = []
    sequences_train = []
    gammas_test = []
    sequences_test = []

    names_train = []
    names_test = []

    jsonTrain = {}
    jsonTest = {}

    RNA_LIST_TRAIN = {i: name for i, name in enumerate(
        listdir(train_data_path))}
    RNA_LIST_TEST = {i: name for i, name in enumerate(
        listdir(test_data_path))}

    for elem in RNA_LIST_TRAIN:
        df = pd.read_csv(join(train_data_path, RNA_LIST_TRAIN[elem]))
        sequence = np.array(df["base"].str[-1])
        gamma = np.array(df["gamma"])
        # We don't want something else in sequences, cause it's RNA
        if ("T" not in sequence and "P" not in sequence):
            sequences_train.append(sequence)
            gammas_train.append(gamma)
            names_train.append(df["name"][0])

    for elem in RNA_LIST_TEST:
        df = pd.read_csv(join(test_data_path, RNA_LIST_TEST[elem]))
        sequence = np.array(df["base"].str[-1])
        gamma = np.array(df["gamma"])
        # We don't want something else in sequences, cause it's RNA
        if ("T" not in sequence and "P" not in sequence):
            sequences_test.append(sequence)
            gammas_test.append(gamma)
            names_test.append(df["name"][0])

    for i in range(len(names_train)):
        jsonTrain[names_train[i]] = {"angles": {
            "gamma": list(gammas_train[i])}, "sequence": "".join(sequences_train[i])}

    for i in range(len(names_test)):
        jsonTest[names_test[i]] = {"angles": {
            "gamma": list(gammas_test[i])}, "sequence": "".join(sequences_test[i])}

    with open('../data/json/train.json', 'w') as fp:
        json.dump(jsonTrain, fp)

    with open('../data/json/test.json', 'w') as fp:
        json.dump(jsonTest, fp)

    return sequences_train, gammas_train, sequences_test, gammas_test
