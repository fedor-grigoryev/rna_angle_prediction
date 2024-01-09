import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from math import isnan
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
from IPython import display
from utils import NucleotideDataset


num_epochs = 45


def custom_mae_loss(output, target):
    # Calculate the absolute difference
    abs_diff = torch.abs(output - target)

    # Calculate the custom MAE
    custom_mae = torch.min(abs_diff, 360 - abs_diff)

    # Calculate the mean
    loss = custom_mae.mean()
    return loss


def train_regressor(model,
                    sequences_train,
                    gammas_train,
                    ):

    torch.tensor(sequences_train)

    train_loader = DataLoader(NucleotideDataset(torch.tensor(sequences_train),
                                                torch.tensor(
                                                    gammas_train)),
                              batch_size=32,
                              shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for _sequences, _angles in train_loader:
            # Forward pass
            outputs = model(_sequences)

            outputs = outputs.squeeze()  # Adjust dimensions if necessary
            # Calculate custom loss
            loss = custom_mae_loss(outputs, _angles)

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


def train_classifier(
        model,
        num_classes,
        sequences_train,
        gammas_train):

    # Addressing class imbalance
    # Flatten the list of class labels
    all_labels = [
        label for seq in gammas_train for label in seq]

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # DataLoader
    train_loader = DataLoader(NucleotideDataset(torch.tensor(sequences_train),
                                                torch.tensor(
                                                    gammas_train)),
                              batch_size=32,
                              shuffle=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        # Training Loop
        for _sequences, _angle_classes in train_loader:
            # Outputs shape: [batch_size, sequence_length, num_classes]
            outputs = model(_sequences)

            # Reshape outputs and angle_classes to use CrossEntropyLoss
            # Reshape to [batch_size * sequence_length, num_classes]
            outputs = outputs.view(-1, num_classes)
            # Flatten to [batch_size * sequence_length]
            _angle_classes = _angle_classes.view(-1)

            # Convert angle_classes to long type
            _angle_classes = _angle_classes.long()

            loss = criterion(outputs, _angle_classes)
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
