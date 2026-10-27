import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""預測模型定義 (PyTorch)。

主模型 (論文): 雙層 LSTM + Dropout + Attention (非雙向, 因股價為嚴格時序)。
基線: 單層 LSTM、RNN、CNN。
"""
import torch
import torch.nn as nn


class LSTMAttention(nn.Module):
    """雙層 LSTM + Attention (論文 Figure 1 架構)。"""

    def __init__(self, n_features, hidden1=64, hidden2=32, dropout=0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.attn_w = nn.Linear(hidden2, hidden2, bias=False)
        self.attn_u = nn.Parameter(torch.randn(hidden2))
        self.fc = nn.Linear(hidden2, 1)

    def attention(self, h):
        u = torch.tanh(self.attn_w(h))          # (B, T, H)
        scores = torch.matmul(u, self.attn_u)   # (B, T)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (h * alpha).sum(dim=1)        # (B, H)
        return context

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out)
        context = self.attention(out)
        return self.fc(context).squeeze(-1)


class SimpleLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class SimpleRNN(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class CNNModel(nn.Module):
    def __init__(self, n_features, window=20, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(32 * window, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.drop(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x).squeeze(-1)


def build_model(model_name: str, n_features: int, window: int = 20):
    if model_name == "lstm_attention":
        return LSTMAttention(n_features, hidden1=64, hidden2=32, dropout=0.1)
    if model_name == "lstm":
        return SimpleLSTM(n_features, hidden=64, dropout=0.1)
    if model_name == "rnn":
        return SimpleRNN(n_features, hidden=64, dropout=0.1)
    if model_name == "cnn":
        return CNNModel(n_features, window=window, dropout=0.1)
    raise ValueError(f"未知模型: {model_name}")