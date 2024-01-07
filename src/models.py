import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim):
        super(LSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # Outputs one angle per time step
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Shape: [batch_size, sequence_length, embedding_dim]
        embedded = self.embedding(x)
        # Shape: [batch_size, sequence_length, hidden_dim]
        lstm_out, _ = self.lstm(embedded)
        # Apply linear layer to each time step
        sequence_length = lstm_out.shape[1]
        out = torch.zeros(x.shape[0], sequence_length, 1).to(embedded.device)
        for i in range(sequence_length):
            out[:, i, :] = self.linear(lstm_out[:, i, :])
        # Removing the last dimension to get [batch_size, sequence_length]
        out = out.squeeze(-1)
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # Outputting 2 classes
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        sequence_length = lstm_out.shape[1]
        out = torch.zeros(x.shape[0], sequence_length,
                          self.num_classes).to(embedded.device)
        for i in range(sequence_length):
            out[:, i, :] = self.linear(lstm_out[:, i, :])
        return out
