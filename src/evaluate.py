import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_classifier(model, test_loader, num_classes):

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
