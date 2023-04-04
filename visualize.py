from typing import List

import matplotlib.pyplot as plt
import numpy as np


def visualize_frank_wolfe(distributions: List[np.array], labels: np.array):
    eps = 0.0001
    for distribution in distributions:
        if abs(sum(distribution) - 1) > eps:
            raise Exception("Each distribution must sum to 1.")
    n = distributions[0].shape[0]
    vertices = np.array([(np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)) for k in range(n)])
    weights = np.array(distributions)

    weighted_vertices = weights @ vertices
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])

    polygon = plt.Polygon(vertices, fill=False)
    ax.add_patch(polygon)
    for i, label in enumerate(labels):
        ax.text(vertices[i, 0], vertices[i, 1], f'{label}', fontsize=12)
    ax.plot(weighted_vertices[:, 0], weighted_vertices[:, 1], c='gray')
    ax.scatter(weighted_vertices[-1, 0], weighted_vertices[-1, 1], c='red')
    plt.axis('off')
    plt.show()
