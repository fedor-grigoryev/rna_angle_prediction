import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import NucleotideDataset, decode_sequences
from utils import encode_sequences

import json


def predict_angles(model,
                   sequences,):
    sequences, _ = encode_sequences(sequences, [])

    with torch.no_grad():
        model.eval()
        sequences = torch.tensor(sequences)

        output_predictions = model(sequences)

        return output_predictions


def evaluate_classifier(model,
                        num_classes,
                        sequences_test,
                        gammas_test,
                        ):

    test_loader = DataLoader(NucleotideDataset(torch.tensor(sequences_test),
                                               torch.tensor(
                                                   gammas_test)))

    # Variables to track metrics
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient tracking
        for _sequences, _angle_classes in test_loader:
            outputs = model(_sequences)  # Get model predictions

            # Reshape outputs for evaluation
            # Shape: [batch_size * sequence_length, num_classes]
            outputs = outputs.view(-1, num_classes)
            # Shape: [batch_size * sequence_length]
            _angle_classes = _angle_classes.view(-1)

            # Convert model outputs to class predictions
            predictions = torch.argmax(outputs, dim=1)

            # Calculate metrics
            accuracies.append(accuracy_score(
                _angle_classes.cpu(), predictions.cpu()))
            precisions.append(precision_score(_angle_classes.cpu(
            ), predictions.cpu(), average='weighted', zero_division=0))
            recalls.append(recall_score(_angle_classes.cpu(),
                                        predictions.cpu(), average='weighted', zero_division=0))
            f1s.append(f1_score(_angle_classes.cpu(),
                                predictions.cpu(), average='weighted', zero_division=0))

    # Average the metrics over all batches
    average_accuracy = sum(accuracies) / len(accuracies)
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    average_f1 = sum(f1s) / len(f1s)

    print(f"Test Accuracy: {average_accuracy:.4f}")
    print(f"Test Precision: {average_precision:.4f}")
    print(f"Test Recall: {average_recall:.4f}")
    print(f"Test F1 Score: {average_f1:.4f}")


def compare_spot_rna_1d_regressor(model,
                                  spot_rna_gammas_train,
                                  sequences_train,
                                  spot_rna_gammas_test,
                                  sequences_test,
                                  ):
    model.eval()
    sequences_train = torch.tensor(sequences_train)

    sequences_test = torch.tensor(sequences_test)

    model_gammas_train = {}
    model_gammas_test = {}

    mae_train = {}
    mae_test = {}

    decoded_sequences_train = decode_sequences(
        sequences_train)
    decoded_sequences_test = decode_sequences(sequences_test)

    with torch.no_grad():
        output_train = model(sequences_train)
        for i in range(len(decoded_sequences_train)):
            model_gammas_train[decoded_sequences_train[i]
                               ] = output_train[i]
        output_test = model(sequences_test)
        for i in range(len(decoded_sequences_test)):
            model_gammas_test[decoded_sequences_test[i]
                              ] = output_test[i]

    for seq in spot_rna_gammas_train.keys():
        if seq in model_gammas_train.keys():
            abs_diff = torch.abs(
                model_gammas_train[seq] - torch.tensor(spot_rna_gammas_train[seq]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_train[seq] = mae.mean().item()

    for seq in spot_rna_gammas_test.keys():
        if seq in model_gammas_test.keys():
            abs_diff = torch.abs(
                model_gammas_test[seq] - torch.tensor(spot_rna_gammas_test[seq]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_test[seq] = mae.mean().item()

    with open("../results/Regressor/mae_train.json", "w") as f:
        json.dump(mae_train, f)

    with open("../results/Regressor/mae_test.json", "w") as f:
        json.dump(mae_test, f)
