from visualisation import visualise_distr
from data_extraction import train_test_extraction
from data_preprocessing import process_data
import torch
import torch.optim as optim
import json


from models import LSTMRegressor, LSTMClassifier
from data_preprocessing import calculate_class_index
from train import train_regressor, train_classifier
from evaluate import evaluate_classifier, compare_spot_rna_1d_regressor

if __name__ == "__main__":
    # visualise_distr("../data/angles/TrainingSet")
    sequences_train, gammas_train, sequences_test, gammas_test = train_test_extraction(
        "../data/angles/TrainingSet", "../data/angles/TestSet")

    padded_sequences_train, padded_gammas_train, masks_train = process_data(
        sequences_train, gammas_train)
    padded_sequences_test, padded_gammas_test, masks_test = process_data(
        sequences_test, gammas_test)

    # Convert gammas to binary classes
    padded_gammas_bin_classes_train = [
        [calculate_class_index(x, num_classes=2) for x in seq] for seq in padded_gammas_train]
    padded_gammas_bin_classes_test = [
        [calculate_class_index(x, num_classes=2) for x in seq] for seq in padded_gammas_test]

    # Convert gammas to multi classes
    padded_gammas_20_classes_train = [
        [calculate_class_index(x, num_classes=20) for x in seq] for seq in padded_gammas_train]
    padded_gammas_20_classes_test = [
        [calculate_class_index(x, num_classes=20) for x in seq] for seq in padded_gammas_test]

    padded_gammas_30_classes_train = [
        [calculate_class_index(x, num_classes=30) for x in seq] for seq in padded_gammas_train]
    padded_gammas_30_classes_test = [
        [calculate_class_index(x, num_classes=30) for x in seq] for seq in padded_gammas_test]

# ------------------ TRAINING AND SAVING MODELS --------------------------------
    # # # REGRESSION APPROACH
    # print("REGRESSION APPROACH")
    # Regressor = LSTMRegressor(
    #     num_embeddings=5, embedding_dim=10, hidden_dim=50)

    # train_regressor(Regressor,
    #                 padded_sequences_train,
    #                 padded_gammas_train,
    #                 masks_train)
    # # Save the model
    # torch.save(Regressor, "../models/Regressor.pt")

    # # BINARY CLASSIFICATION APPROACH
    # print("BINARY CLASSIFICATION APPROACH")

    # BinClassifier = LSTMClassifier(
    #     num_embeddings=5, embedding_dim=10, hidden_dim=50, num_classes=2)

    # train_classifier(BinClassifier,
    #                  num_classes=2,
    #                  padded_sequences_train=padded_sequences_train,
    #                  padded_gammas_train=padded_gammas_bin_classes_train,
    #                  masks_train=masks_train,
    #                  )

    # # Save the model
    # torch.save(BinClassifier, "../models/BinClassifier.pt")

    # # MULTI-CLASS CLASSIFICATION APPROACH
    # # 20 Classes
    # print("20 CLASSES CLASSIFICATION APPROACH")
    # TwentyClassifier = LSTMClassifier(
    #     num_embeddings=5, embedding_dim=10, hidden_dim=50, num_classes=20)

    # train_classifier(TwentyClassifier,
    #                  num_classes=20,
    #                  padded_sequences_train=padded_sequences_train,
    #                  padded_gammas_train=padded_gammas_20_classes_train,
    #                  masks_train=masks_train,
    #                  )

    # # Save the model
    # torch.save(TwentyClassifier, "../models/TwentyClassifier.pt")

    # print("30 CLASSES CLASSIFICATION APPROACH")
    # ThirtyClassifier = LSTMClassifier(
    #     num_embeddings=5, embedding_dim=10, hidden_dim=50, num_classes=30)

    # train_classifier(ThirtyClassifier,
    #                  num_classes=30,
    #                  padded_sequences_train=padded_sequences_train,
    #                  padded_gammas_train=padded_gammas_30_classes_train,
    #                  masks_train=masks_train,
    #                  )

    # # Save the model
    # torch.save(ThirtyClassifier, "../models/ThirtyClassifier.pt")

# ------------------ LOADING MODELS AND EVALUATING -----------------------------

    # Load json files

    f_spot_rna_angles_train = open("../data/SPOT-RNA-1D/training.json")
    f_spot_rna_angles_test = open("../data/SPOT-RNA-1D/test.json")
    spot_rna_angles_train = json.load(f_spot_rna_angles_train)
    spot_rna_angles_test = json.load(f_spot_rna_angles_test)
    f_spot_rna_angles_train.close()
    f_spot_rna_angles_test.close()

    spot_rna_gammas_train = {}
    spot_rna_gammas_test = {}

    for key in spot_rna_angles_train.keys():
        spot_rna_gammas_train[spot_rna_angles_train[key]
                              ['sequence']] = spot_rna_angles_train[key]['angles']['gamma']

    for key in spot_rna_angles_test.keys():
        spot_rna_gammas_test[spot_rna_angles_test[key]
                             ['sequence']] = spot_rna_angles_test[key]['angles']['gamma']

    # # REGRESSION APPROACH
    # print("REGRESSION APPROACH")
    # Regressor = torch.load("../models/Regressor.pt")
    # compare_spot_rna_1d_regressor(Regressor,
    #                               spot_rna_gammas_train,
    #                               padded_sequences_train,
    #                               masks_train,
    #                               spot_rna_gammas_test,
    #                               padded_sequences_test,
    #                               masks_test)

    # # BINARY CLASSIFICATION APPROACH
    # print("BINARY CLASSIFICATION APPROACH")
    # BinClassifier = torch.load("../models/BinClassifier.pt")

    # evaluate_classifier(BinClassifier,
    #                     num_classes=2,
    #                     padded_sequences_test=padded_sequences_test,
    #                     padded_gammas_test=padded_gammas_bin_classes_test,
    #                     masks_test=masks_test)

    # # MULTI-CLASS CLASSIFICATION APPROACH
    # # 20 Classes
    # print("20 CLASSES CLASSIFICATION APPROACH")
    # TwentyClassifier = torch.load("../models/TwentyClassifier.pt")

    # evaluate_classifier(TwentyClassifier,
    #                     num_classes=20,
    #                     padded_sequences_test=padded_sequences_test,
    #                     padded_gammas_test=padded_gammas_20_classes_test,
    #                     masks_test=masks_test)

    # # 30 Classes
    # print("30 CLASSES CLASSIFICATION APPROACH")
    # ThirtyClassifier = torch.load("../models/ThirtyClassifier.pt")

    # evaluate_classifier(ThirtyClassifier,
    #                     num_classes=30,
    #                     padded_sequences_test=padded_sequences_test,
    #                     padded_gammas_test=padded_gammas_30_classes_test,
    #                     masks_test=masks_test)
