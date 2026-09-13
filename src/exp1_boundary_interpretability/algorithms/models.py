from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SwiGLUFFN(nn.Module):
    """Residual SwiGLU block used by the released Elastic Time predictor."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.value = nn.Linear(dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(values)
        update = self.output(F.silu(self.gate(normalized)) * self.value(normalized))
        return values + update


class ElasticPredictor(nn.Module):
    """Open-loop GRUCell predictor following the released Elastic Time design."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.initial_hidden = nn.Parameter(torch.zeros(hidden_dim))
        self.output = nn.Sequential(
            SwiGLUFFN(hidden_dim, hidden_dim * 4),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, frames: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dimension {self.feature_dim}, got {frames.shape[-1]}")
        leading_shape = frames.shape[:-1]
        flattened = frames.reshape(-1, self.feature_dim)
        count = flattened.shape[0]
        if state is None:
            state = self.initial_hidden.view(1, 1, self.hidden_dim).expand(self.num_layers, count, -1)
        elif state.shape != (self.num_layers, count, self.hidden_dim):
            raise ValueError(
                f"state must have shape {(self.num_layers, count, self.hidden_dim)}, got {tuple(state.shape)}"
            )

        current = self.input_projection(flattened)
        next_state = []
        for layer, cell in enumerate(self.cells):
            current = cell(current, state[layer])
            next_state.append(current)
        prediction = self.output(current).reshape(*leading_shape, self.feature_dim)
        return prediction, torch.stack(next_state)

    def rollout(self, anchor: torch.Tensor, steps: int) -> torch.Tensor:
        if steps <= 0:
            return anchor.new_empty((anchor.shape[0], 0, anchor.shape[-1]))
        current = anchor
        state = None
        outputs = []
        for _ in range(steps):
            current, state = self.forward(current, state)
            outputs.append(current)
        return torch.stack(outputs, dim=1)

    @torch.inference_mode()
    def segment_costs(self, features: np.ndarray, max_span: int, device: str) -> np.ndarray:
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        length = len(features)
        costs = np.full((length, max_span + 1), np.inf, dtype=np.float64)
        costs[:, 1] = 0.0
        self.eval()
        steps = min(max_span - 1, length - 1)
        if steps <= 0:
            return costs
        prediction = self.rollout(x[:-1], steps)
        errors = prediction.new_zeros((length - 1, steps))
        for step in range(1, steps + 1):
            valid = length - step
            errors[:valid, step - 1] = (prediction[:valid, step - 1] - x[step:]).pow(2).mean(dim=-1)
        cumulative = torch.cumsum(errors, dim=1).cpu().numpy()
        for size in range(2, steps + 2):
            valid = length - size + 1
            costs[:valid, size] = cumulative[:valid, size - 2]
        return costs


class LocalPredictabilityModel(nn.Module):
    """Predict a hidden center frame from local left/right neighbors only."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, radius: int = 2) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.radius = radius
        self.neighbor_projection = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU())
        self.predictor = nn.Sequential(
            nn.Linear(2 * radius * hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, feature_dim),
        )

    def contexts(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [B, T, D]")
        batch, length, dim = sequence.shape
        padded = F.pad(sequence.transpose(1, 2), (self.radius, self.radius)).transpose(1, 2)
        neighbors = []
        for offset in range(-self.radius, 0):
            begin = self.radius + offset
            neighbors.append(padded[:, begin : begin + length])
        for offset in range(1, self.radius + 1):
            begin = self.radius + offset
            neighbors.append(padded[:, begin : begin + length])
        projected = [self.neighbor_projection(item) for item in neighbors]
        return torch.cat(projected, dim=-1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.contexts(sequence))

    @torch.inference_mode()
    def difficulty(self, features: np.ndarray, device: str) -> np.ndarray:
        self.eval()
        x = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        prediction = self.forward(x)
        mse = (prediction - x).pow(2).mean(dim=-1)
        cosine = 1.0 - F.cosine_similarity(prediction, x, dim=-1)
        score = (mse + 0.1 * cosine).squeeze(0).cpu().numpy().astype(np.float32)
        invalid = min(self.radius, len(score))
        score[:invalid] = 0.0
        score[len(score) - invalid :] = 0.0
        return score


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_frame = (prediction - target).pow(2).mean(dim=-1)
    return (per_frame * mask).sum() / mask.sum().clamp_min(1.0)
