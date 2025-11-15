"""
Sequence model (GRU) for churn early-warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class EventSequenceDataset(Dataset):
    """
    Torch dataset wrapping padded event sequences for each user.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        lengths: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


class GRUChurnModel(nn.Module):
    """
    GRU network predicting churn probability from event sequences.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:  # type: ignore[override]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        logits = self.classifier(hidden[-1])
        return logits.squeeze(-1)


@dataclass
class ChurnEarlyWarning:
    """
    Wraps training/inference for the GRU churn model.
    """

    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 1
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    device: str = "cpu"

    model: Optional[GRUChurnModel] = None

    def fit(
        self,
        train_sequences: np.ndarray,
        train_lengths: np.ndarray,
        train_labels: np.ndarray,
    ) -> None:
        dataset = EventSequenceDataset(train_sequences, train_lengths, train_labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = GRUChurnModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_len, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_len = batch_len.to(self.device)
                batch_y = batch_y.to(self.device)

                optimiser.zero_grad()
                preds = self.model(batch_x, batch_len)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch_y)

            avg_loss = epoch_loss / len(dataset)
            print(f"[ChurnEarlyWarning] epoch={epoch+1}/{self.epochs} loss={avg_loss:.4f}")

    def predict_proba(
        self,
        sequences: np.ndarray,
        lengths: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        dataset = EventSequenceDataset(sequences, lengths, np.zeros(len(lengths)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_len, _ in loader:
                batch_x = batch_x.to(self.device)
                batch_len = batch_len.to(self.device)
                prob = self.model(batch_x, batch_len).cpu().numpy()
                preds.append(prob)
        return np.concatenate(preds, axis=0)
