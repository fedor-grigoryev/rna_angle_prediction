from visualisation import visualise_distr
from data_extraction import train_test_extraction
from data_preprocessing import process_data
import torch
import torch.optim as optim


from models import LSTMRegressor, LSTMClassifier
from data_preprocessing import calculate_class_index
from train import train_regressor, train_classifier
from evaluate import evaluate_classifier

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
    padded_gammas_multi_classes_train = [
        [calculate_class_index(x, num_classes=20) for x in seq] for seq in padded_gammas_train]
    padded_gammas_multi_classes_test = [
        [calculate_class_index(x, num_classes=20) for x in seq] for seq in padded_gammas_test]

    # REGRESSION APPROACH
    # print("REGRESSION APPROACH")
    # Regressor = LSTMRegressor(num_embeddings=4, embedding_dim=10, hidden_dim=50)

    # train_regressor(Regressor,
    #                 padded_sequences_train,
    #                 padded_gammas_train,
    #                 masks_train)

    # BINARY CLASSIFICATION APPROACH
    print("BINARY CLASSIFICATION APPROACH")

    BinClassifier = LSTMClassifier(
        num_embeddings=4, embedding_dim=10, hidden_dim=50, num_classes=2)

    train_classifier(BinClassifier,
                     num_classes=2,
                     padded_sequences_train=padded_sequences_train,
                     padded_gammas_train=padded_gammas_bin_classes_train,
                     masks_train=masks_train,
                     )

    evaluate_classifier(BinClassifier,
                        num_classes=2,
                        padded_sequences_test=padded_sequences_test,
                        padded_gammas_test=padded_gammas_bin_classes_test,
                        masks_test=masks_test)
    # MULTI-CLASS CLASSIFICATION APPROACH
    print("MULTI-CLASS CLASSIFICATION APPROACH")
    MultiClassifier = LSTMClassifier(
        num_embeddings=4, embedding_dim=10, hidden_dim=50, num_classes=20)

    train_classifier(MultiClassifier,
                     num_classes=20,
                     padded_sequences_train=padded_sequences_train,
                     padded_gammas_train=padded_gammas_multi_classes_train,
                     masks_train=masks_train,
                     )

    evaluate_classifier(MultiClassifier,
                        num_classes=20,
                        padded_sequences_test=padded_sequences_test,
                        padded_gammas_test=padded_gammas_multi_classes_test,
                        masks_test=masks_test)
