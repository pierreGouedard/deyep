# Global import
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_img(title: str, ax_img: np.array, ax_compare_img: Optional[np.array] = None) -> None:
    if ax_compare_img is not None:
        _, axis = plt.subplots(1, 2, figsize=(20, 10))

        axis[0].imshow(ax_img)
        axis[1].imshow(ax_compare_img)
        plt.title(title)
        plt.show()
    else:
        plt.figure(figsize=(20, 10))
        plt.imshow(ax_img)
        plt.title(title)
        plt.show()


def bar_plot(title: str, ax_bars: np.array) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(list(map(str, range(len(ax_bars)))), ax_bars)
    plt.title(title)
    plt.show()
