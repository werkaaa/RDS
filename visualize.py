from typing import List, Tuple

import imageio
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


def draw_axis(ax, point1: np.array, point2: np.array, xytext: Tuple, ticks: int = 10):
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    intercept = point1[1] - slope * point1[0]

    x_values = np.linspace(point1[0], point2[0], num=ticks + 1)
    y_values = slope * x_values + intercept

    labels = [f'{i / ticks}' for i in range(ticks + 1)]

    for i, (x, y) in enumerate(zip(x_values, y_values)):
        ax.plot(x, y, '|', label=labels[i], c='black')
        ax.annotate(labels[i], xy=(x, y), xytext=xytext, textcoords='offset points')


def plot_ternary(distributions: List[np.array], Fs: List[callable] = None):
    vertices = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3) / 2]
    ])

    x_scaled = []
    y_scaled = []
    for distribution in distributions:
        a, b, c = distribution
        x_scaled.append(b + c / 2)
        y_scaled.append(np.sqrt(3) / 2 * c)

    fig, axs = plt.subplots(1, len(Fs), figsize=(6 * len(Fs), 5))

    for i, ax in enumerate(axs):
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)

        F = Fs[i]
        if F is not None:
            delta = 0.005
            X = np.arange(0.0, 1.0, delta)
            Y = np.arange(0.0, np.sqrt(3) / 2, delta)
            Z = []
            for y in Y:
                z_row = []
                for x in X:
                    c = 2 / np.sqrt(3) * y
                    b = x - y / np.sqrt(3)
                    a = 1 - b - c
                    if b < 0 or a < 0:
                        z_row.append(None)
                    else:
                        z_row.append(F(np.array([a, b, c])))
                Z.append(z_row)
            Z = np.array(Z, dtype=float)

            CS = ax.contourf(X, Y, np.minimum(Z, 50), 50)

        ax.plot(
            vertices[:, 0].tolist() + [vertices[0, 0]],
            vertices[:, 1].tolist() + [vertices[0, 1]],
            color='black')

        draw_axis(ax, vertices[0], vertices[1], (0, -20))
        draw_axis(ax, vertices[1], vertices[2], (10, 0))
        draw_axis(ax, vertices[2], vertices[0], (-20, 0))

        labels = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        xytext = [(-30, -30), (30, -30), (0, 30)]
        for j, label in enumerate(labels):
            ax.annotate(
                text=f'{label}',
                xy=tuple(vertices[j]),
                xytext=xytext[j],
                textcoords='offset points',
                ha='center')

        ax.plot(x_scaled, y_scaled, color='grey')
        ax.scatter(x_scaled[-1], y_scaled[-1], color='red', zorder=2)

        ax.axis('off')

    plt.savefig(f'./gif/img_{len(distributions)}.png',
                transparent=False,
                facecolor='white')
    plt.close()


def generate_simplex_gif(distributions: List[np.array], path: str = './gif/example.gif', F1=None, F2=None):
    if type(F1) != list:
        F1 = [F1] * len(F2)
    for t in range(len(distributions)):
        plot_ternary(distributions[:t + 1], [F1[t], F2[t]])

    frames = []
    for t in range(len(distributions)):
        image = imageio.v2.imread(f'./gif/img_{t + 1}.png')
        frames.append(image)

    imageio.mimsave(path,
                    frames,
                    duration=300,  # Values between 100 and 500 make sense
                    loop=0)
