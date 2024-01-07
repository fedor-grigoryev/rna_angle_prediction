import torch
import numpy as np
from math import isnan
import matplotlib.pyplot as plt
from IPython import display


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


def train_regressor(model, train_loader, optimizer):
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

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += 0 if np.isnan(loss.item()) else loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")


def train_classifier(model, train_loader, criterion, optimizer, num_classes):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

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

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Loops')
        plt.ylabel('Loss')
        display.clear_output(wait=True)
        display.display(plt.show())
        display.clear_output(wait=True)
