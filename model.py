import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.input_size = 128
        self.hidden_state_size = 128
        self.embedding_dim = 128
        self.num_lstm_layers = 4

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_state_size,
            num_layers=self.num_lstm_layers,
            dropout=0.1,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_state_size, 256),
            nn.Linear(256, n_vocab)
        )

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_lstm_layers, sequence_length, self.hidden_state_size),
                torch.zeros(self.num_lstm_layers, sequence_length, self.hidden_state_size))