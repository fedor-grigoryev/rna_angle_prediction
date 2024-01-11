import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import pad_sequences, encode_sequences, convert_classes_to_angles, NucleotideDataset, decode_sequences

import json


def predict_angles(model,
                   sequences,
                   num_classes=None):
    sequences, _ = encode_sequences(sequences, [])
    padded_sequences, _, _ = pad_sequences(sequences, sequences, maxlen=200)

    with torch.no_grad():
        model.eval()
        padded_sequences = torch.tensor(padded_sequences)

        output_predictions = model(padded_sequences)

        if num_classes is not None:
            output_predictions = torch.argmax(output_predictions, dim=2)
            output_predictions = output_predictions.apply_(lambda class_index: class_index *
                                                           360/num_classes + 360/num_classes/2)

        return output_predictions


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
            ), predictions.cpu(), average='weighted', zero_division=1))
            recalls.append(recall_score(angle_classes_masked.cpu(),
                                        predictions.cpu(), average='weighted', zero_division=1))
            f1s.append(f1_score(angle_classes_masked.cpu(),
                                predictions.cpu(), average='weighted', zero_division=1))

    # Average the metrics over all batches
    average_accuracy = sum(accuracies) / len(accuracies)
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    average_f1 = sum(f1s) / len(f1s)

    print(f"Test Accuracy: {average_accuracy:.4f}")
    print(f"Test Precision: {average_precision:.4f}")
    print(f"Test Recall: {average_recall:.4f}")
    print(f"Test F1 Score: {average_f1:.4f}")


def compare_dssr_regressor(model,
                           dssr_gammas_train,
                           padded_sequences_train,
                           masks_train,
                           dssr_gammas_test,
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

    for seq in dssr_gammas_train.keys():
        shinked_seq = seq[:200]
        if shinked_seq in model_gammas_train.keys():
            abs_diff = torch.abs(
                model_gammas_train[shinked_seq] - torch.tensor(dssr_gammas_train[seq][:200]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_train[seq] = mae.mean().item()

    for seq in dssr_gammas_test.keys():
        shinked_seq = seq[:200]
        if shinked_seq in model_gammas_test.keys():
            abs_diff = torch.abs(
                model_gammas_test[shinked_seq] - torch.tensor(dssr_gammas_test[seq][:200]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_test[seq] = mae.mean().item()

    with open("../results/Regressor/mae_train.json", "w") as f:
        json.dump(mae_train, f)

    with open("../results/Regressor/mae_test.json", "w") as f:
        json.dump(mae_test, f)


def compare_dssr_classifier(model,
                            num_classes,
                            dssr_gammas_train,
                            padded_sequences_train,
                            masks_train,
                            dssr_gammas_test,
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
        output_train = torch.argmax(output_train, dim=2)
        output_train = output_train.apply_(
            lambda class_index: class_index * 360/num_classes + 360/num_classes/2)

        # print('output_train')
        # print(output_train)

        for i in range(len(decoded_sequences_train)):
            model_gammas_train[decoded_sequences_train[i]
                               ] = output_train[i][:int(sum(masks_train[i]))]

        output_test = model(sequences_test)
        output_test = torch.argmax(output_test, dim=2)
        output_test.apply_(lambda class_index: class_index *
                           360/num_classes + 360/num_classes/2)

        for i in range(len(decoded_sequences_test)):
            model_gammas_test[decoded_sequences_test[i]
                              ] = output_test[i][:int(sum(masks_test[i]))]

    for seq in dssr_gammas_train.keys():
        shinked_seq = seq[:200]
        if shinked_seq in model_gammas_train.keys():
            abs_diff = torch.abs(
                model_gammas_train[shinked_seq] - torch.tensor(dssr_gammas_train[seq][:200]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_train[seq] = mae.mean().item()

    for seq in dssr_gammas_test.keys():
        shinked_seq = seq[:200]
        if shinked_seq in model_gammas_test.keys():
            abs_diff = torch.abs(
                model_gammas_test[shinked_seq] - torch.tensor(dssr_gammas_test[seq][:200]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_test[seq] = mae.mean().item()

    with open(f'../results/{num_classes}ClassClassifier/mae_train.json', "w") as f:
        json.dump(mae_train, f)

    with open(f'../results/{num_classes}ClassClassifier/mae_test.json', "w") as f:
        json.dump(mae_test, f)


def compare_dssr_bin_stats_classifier(model,
                                      dssr_gammas_train,
                                      padded_sequences_train,
                                      masks_train,
                                      dssr_gammas_test,
                                      padded_sequences_test,
                                      masks_test,
                                      ):
    # Define a mapping from classes to angle values
    class_to_angle_mapping = {
        0: 55.038,
        1: 189.917,
        # Add more mappings if needed for other classes
    }

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
        output_train = torch.argmax(output_train, dim=2)
        output_train = output_train.apply_(
            lambda class_index: class_to_angle_mapping[class_index])

        print('output_train')
        print(output_train)

        for i in range(len(decoded_sequences_train)):
            model_gammas_train[decoded_sequences_train[i]
                               ] = output_train[i][:int(sum(masks_train[i]))]

        output_test = model(sequences_test)
        output_test = torch.argmax(output_test, dim=2)
        output_test.apply_(
            lambda class_index: class_to_angle_mapping[class_index])

        for i in range(len(decoded_sequences_test)):
            model_gammas_test[decoded_sequences_test[i]
                              ] = output_test[i][:int(sum(masks_test[i]))]

    for seq in dssr_gammas_train.keys():
        shinked_seq = seq[:200]
        if shinked_seq in model_gammas_train.keys():
            abs_diff = torch.abs(
                model_gammas_train[shinked_seq] - torch.tensor(dssr_gammas_train[seq][:200]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_train[seq] = mae.mean().item()

    for seq in dssr_gammas_test.keys():
        shinked_seq = seq[:200]
        if shinked_seq in model_gammas_test.keys():
            abs_diff = torch.abs(
                model_gammas_test[shinked_seq] - torch.tensor(dssr_gammas_test[seq][:200]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_test[seq] = mae.mean().item()

    with open(f'../results/BinStatsClassClassifier/mae_train.json', "w") as f:
        json.dump(mae_train, f)

    with open(f'../results/BinStatsClassClassifier/mae_test.json', "w") as f:
        json.dump(mae_test, f)


def compare_dssr_spot_angles(
        dssr_gammas_train,
        spot_rna_gammas_train,
        dssr_gammas_test,
        spot_rna_gammas_test,
):
    mae_train = {}
    mae_test = {}

    for seq in dssr_gammas_train.keys():
        if seq in spot_rna_gammas_train.keys():
            abs_diff = torch.abs(
                torch.tensor(spot_rna_gammas_train[seq]) - torch.tensor(dssr_gammas_train[seq]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_train[seq] = mae.mean().item()

    for seq in dssr_gammas_test.keys():
        if seq in spot_rna_gammas_test.keys():
            abs_diff = torch.abs(
                torch.tensor(spot_rna_gammas_test[seq]) - torch.tensor(dssr_gammas_test[seq]))
            mae = torch.min(abs_diff, 360 - abs_diff)
            mae_test[seq] = mae.mean().item()

    with open(f'../results/SPOT-RNA-1D/mae_train.json', "w") as f:
        json.dump(mae_train, f)

    with open(f'../results/SPOT-RNA-1D/mae_test.json', "w") as f:
        json.dump(mae_test, f)
