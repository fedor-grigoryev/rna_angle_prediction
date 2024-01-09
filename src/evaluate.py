import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import NucleotideDataset, decode_sequences

import json


def evaluate_classifier(model,
                        num_classes,
                        padded_sequences_test,
                        padded_gammas_test,
                        masks_test):

    test_loader = DataLoader(NucleotideDataset(torch.tensor(padded_sequences_test),
                                               torch.tensor(
                                                   padded_gammas_test),
                                               torch.tensor(masks_test)))

    # Variables to track metrics
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient tracking
        for _sequences, _angle_classes, _masks in test_loader:
            outputs = model(_sequences)  # Get model predictions

            # Reshape outputs for evaluation
            # Shape: [batch_size * sequence_length, num_classes]
            outputs = outputs.view(-1, num_classes)
            # Shape: [batch_size * sequence_length]
            _angle_classes = _angle_classes.view(-1)
            mask_flat = _masks.view(-1).bool()

            # Mask outputs and angle_classes
            outputs_masked = outputs[mask_flat]
            angle_classes_masked = _angle_classes[mask_flat]

            # Convert model outputs to class predictions
            predictions = torch.argmax(outputs_masked, dim=1)

            # Calculate metrics
            accuracies.append(accuracy_score(
                angle_classes_masked.cpu(), predictions.cpu()))
            precisions.append(precision_score(angle_classes_masked.cpu(
            ), predictions.cpu(), average='weighted', zero_division=0))
            recalls.append(recall_score(angle_classes_masked.cpu(),
                                        predictions.cpu(), average='weighted', zero_division=0))
            f1s.append(f1_score(angle_classes_masked.cpu(),
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
                                  padded_sequences_train,
                                  masks_train,
                                  spot_rna_gammas_test,
                                  padded_sequences_test,
                                  masks_test,
                                  ):
    model.eval()
    sequences_train = torch.tensor(padded_sequences_train)
    masks_train = torch.tensor(masks_train)

    sequences_test = torch.tensor(padded_sequences_test)
    masks_test = torch.tensor(masks_test)

    model_gammas_train = {}
    model_gammas_test = {}

    mae_train = {}
    mae_test = {}

    decoded_sequences_train = decode_sequences(
        sequences_train, masks_train)
    decoded_sequences_test = decode_sequences(sequences_test, masks_test)

    with torch.no_grad():
        output_train = model(sequences_train)
        for i in range(len(decoded_sequences_train)):
            model_gammas_train[decoded_sequences_train[i]
                               ] = output_train[i][:int(sum(masks_train[i]))]

        output_test = model(sequences_test)
        for i in range(len(decoded_sequences_test)):
            model_gammas_test[decoded_sequences_test[i]
                              ] = output_test[i][:int(sum(masks_test[i]))]

    for seq in spot_rna_gammas_train.keys():
        shinked_seq = seq[:200]
        if shinked_seq in model_gammas_train.keys():
            abs_diff = torch.abs(
                model_gammas_train[shinked_seq] - torch.tensor(spot_rna_gammas_train[seq][:200]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_train[seq] = mae.mean().item()

    for seq in spot_rna_gammas_test.keys():
        shinked_seq = seq[:200]
        if shinked_seq in model_gammas_test.keys():
            abs_diff = torch.abs(
                model_gammas_test[shinked_seq] - torch.tensor(spot_rna_gammas_test[seq][:200]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_test[seq] = mae.mean().item()

    with open("../results/Regressor/mae_train.json", "w") as f:
        json.dump(mae_train, f)

    with open("../results/Regressor/mae_test.json", "w") as f:
        json.dump(mae_test, f)
