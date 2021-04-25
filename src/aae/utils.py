from typing import List
import numpy as np
import matplotlib.pyplot as plt


def get_hidden_sizes(input_size: int, hidden_size: int, n_layers: int) -> List[int]:
    return list(np.linspace(input_size, hidden_size, n_layers + 1).astype(int))


def plot_latent(latent: np.ndarray, label: np.ndarray) -> plt.figure:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for l in np.unique(label):
        sub_latent = latent[label == l]
        ax.scatter(sub_latent[:, 0], sub_latent[:, 1], label=l)
    return fig
