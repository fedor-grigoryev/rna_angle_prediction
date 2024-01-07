import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from math import isnan
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
from IPython import display
from data_preprocessing import NucleotideDataset
from evaluate import evaluate_classifier


num_epochs = 25


def custom_mae_loss(output, target, mask):
    # Calculate the absolute difference
    abs_diff = torch.abs(output - target)

    # Calculate the custom MAE
    custom_mae = torch.min(abs_diff, 360 - abs_diff)

    # Apply the mask to ignore padded values
    masked_mae = custom_mae * mask  # sometimes returns nans - not sure why

    # Calculate the mean, considering only the non-zero (non-padded) elements
    loss = torch.sum(masked_mae) / torch.sum(mask)
    return loss


def train_regressor(model,
                    padded_sequences_train,
                    padded_gammas_train,
                    masks_train,
                    ):

    train_loader = DataLoader(NucleotideDataset(torch.tensor(padded_sequences_train),
                                                torch.tensor(
                                                    padded_gammas_train),
                                                torch.tensor(masks_train)),
                              batch_size=32,
                              shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for _sequences, _angles, _masks in train_loader:
            # Forward pass
            outputs = model(_sequences)

            # Make sure outputs, angles, and masks are of the same shape
            outputs = outputs.squeeze()  # Adjust dimensions if necessary
            # Calculate custom loss
            loss = custom_mae_loss(outputs, _angles, _masks)

            train_losses.append(loss.item())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += 0 if np.isnan(loss.item()) else loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

        # plt.figure(figsize=(10, 6))
        # plt.plot(train_losses, label='Train Loss')
        # plt.xlabel('Loops')
        # plt.ylabel('Loss')
        # display.clear_output(wait=True)
        # display.display(plt.show())
        # display.clear_output(wait=True)
    return model


def train_classifier(model,
                     num_classes,
                     padded_sequences_train,
                     padded_gammas_train,
                     masks_train,
                     padded_sequences_test,
                     padded_gammas_test,
                     masks_test):
    # Addressing class imbalance
    # Flatten the list of class labels
    all_labels = [
        label for seq in padded_gammas_train for label in seq]

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # If you're using a GPU, send the weights to the same device as your model
    if torch.cuda.is_available():
        class_weights_tensor = class_weights_tensor.to('cuda')

    # DataLoader
    train_loader = DataLoader(NucleotideDataset(torch.tensor(padded_sequences_train),
                                                torch.tensor(
                                                    padded_gammas_train),
                                                torch.tensor(masks_train)),
                              batch_size=32,
                              shuffle=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        # Training Loop
        for _sequences, _angle_classes, _masks in train_loader:
            # Outputs shape: [batch_size, sequence_length, num_classes]
            outputs = model(_sequences)

            # Reshape outputs and angle_classes to use CrossEntropyLoss
            # Reshape to [batch_size * sequence_length, num_classes]
            outputs = outputs.view(-1, num_classes)
            # Flatten to [batch_size * sequence_length]
            _angle_classes = _angle_classes.view(-1)

            # Convert angle_classes to long type
            _angle_classes = _angle_classes.long()

            # Apply mask to ignore the padded positions
            # Flatten the mask and use it to select the non-padded positions
            mask_flat = _masks.view(-1).bool()
            outputs_masked = outputs[mask_flat]

            angle_classes_masked = _angle_classes[mask_flat]

            loss = criterion(outputs_masked, angle_classes_masked)
            train_losses.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += 0 if np.isnan(loss.item()) else loss.item()

        # plt.figure(figsize=(10, 6))
        # plt.plot(train_losses, label='Train Loss')
        # plt.xlabel('Loops')
        # plt.ylabel('Loss')
        # display.clear_output(wait=True)
        # display.display(plt.show())
        # display.clear_output(wait=True)
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
    # Evaluate the model
    evaluate_classifier(model,
                        num_classes=num_classes,
                        padded_sequences_test=padded_sequences_test,
                        padded_gammas_test=padded_gammas_test,
                        masks_test=masks_test)
