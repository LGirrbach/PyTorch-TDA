import torch
import torch.nn as nn

from gudhi_wrappers import Rips
from gudhi_wrappers import Cubical


class RipsLayer(nn.Module):
    def __init__(self, mel=12, dim=1, card=50):
        super().__init__()
        self.mel = mel
        self.dim = dim
        self.card = card

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle batched and unbatched inputs
        x_shape_original = x.shape

        if len(x_shape_original) == 2:  # No batches!
            x = x.unsqueeze(0)
        elif len(x_shape_original) == 3:
            pass
        else:
            raise RuntimeError(f"Input must have 2 or 3 dimensions, but found {len(x_shape_original)}")

        # x: shape [batch x #points x features]

        # Compute Distance Matrix
        distance_matrix = torch.cdist(x, x)
        distance_matrix_numpy = distance_matrix.detach()
        distance_matrix_numpy = distance_matrix_numpy.cpu()
        distance_matrix_numpy = distance_matrix_numpy.numpy()

        ids = []
        ids_mask = []

        # Iterate over batch samples
        batch_size = distance_matrix.shape[0]
        num_points = distance_matrix.shape[2]
        for k in range(batch_size):
            # Compute vertices associated to positive and negative simplices
            ids_k = Rips(distance_matrix_numpy[k], self.mel, self.dim, self.card)
            ids_k = torch.tensor(ids_k).long()
            ids_k = ids_k.reshape(2 * self.card, 2)

            if self.dim == 0:
                ids_k = ids_k[1::2, :]

            ids_k_mask = torch.any(torch.eq(ids_k, -1), dim=-1)
            ids_k[:, 0] = ids_k[:, 0] + k * num_points
            ids.append(ids_k)
            ids_mask.append(ids_k_mask)

        ids = torch.cat(ids, dim=0)
        ids_mask = torch.cat(ids_mask, dim=0)
        # ids: shape [batch * 2 * card, 2]

        # Batch index select trick:
        # Delete batch dim and select from 2d matrix,
        # then reshape
        distance_matrix = distance_matrix.reshape(-1, num_points)
        dgm = distance_matrix[ids[:, 0], ids[:, 1]]
        dgm = torch.masked_fill(dgm, mask=ids_mask, value=0.0)

        # Get persistence diagram by simply picking the corresponding entries in the distance matrix
        if self.dim > 0:
            dgm = dgm.reshape(batch_size, self.card, 2)
        else:
            dgm = dgm.reshape(batch_size, self.card, 1)
            dgm = torch.cat([torch.zeros_like(dgm), dgm], axis=2)

        if len(x_shape_original) == 2:
            dgm = dgm.squeeze(0)

        return dgm


class CubicalLayer(nn.Module):
    def __init__(self, dim=1, card=50):
        super().__init__()
        self.dim = dim
        self.card = card

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Iterate over batch samples
        batch_size = x.shape[0]
        first_data_dim_size = x.shape[1]
        num_data_dims = len(x.shape[1:])

        ids = []
        ids_mask = []

        for k in range(batch_size):
            ids_k = Cubical(x[k].detach().cpu().numpy(), self.dim, self.card)
            ids_k = torch.tensor(ids_k).long()
            ids_k = ids_k.reshape(-1, num_data_dims)
            ids_k_mask = torch.any(torch.eq(ids_k, -1), dim=-1)

            ids_k[:, 0] = ids_k[:, 0] + k * first_data_dim_size
            ids.append(ids_k)
            ids_mask.append(ids_k_mask)

        ids = torch.cat(ids, dim=0)
        ids = torch.chunk(ids, chunks=num_data_dims, dim=1)
        ids_mask = torch.cat(ids_mask, dim=0).unsqueeze(1)

        x = x.reshape((-1, *x.shape[2:]))
        dgm = x[ids]
        dgm = torch.masked_fill(dgm, mask=ids_mask, value=0.0)
        dgm = dgm.reshape(batch_size, self.card, 2)
        return dgm
