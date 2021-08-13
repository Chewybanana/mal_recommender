from fastai.collab import Module, Embedding, sigmoid_range

import torch
import torch.nn as nn


class DotProduct(Module):
    def __init__(self, n_users, n_animes, n_factors, y_range=(0, 10.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.anime_factors = Embedding(n_animes, n_factors)

        self.user_bias = Embedding(n_users, 1)
        self.anime_bias = Embedding(n_animes, 1)

        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors(x[:, 0])
        animes = self.anime_factors(x[:, 1])

        res = (users * animes).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:, 0]) + self.anime_bias(x[:, 1])
        return sigmoid_range(res, *self.y_range)
#         return (users * animes).sum(dim=1)


class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0, 10)):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.y_range = y_range

    def forward(self, x):
        embs = self.user_factors(x[:, 0]), self.item_factors(x[:, 1])
        x = self.layers(torch.cat(embs, dim=1))

        # return x
        # return torch.clamp(x, *self.y_range)
        return sigmoid_range(x, *self.y_range)
