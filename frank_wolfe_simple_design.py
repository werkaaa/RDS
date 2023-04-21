from typing import List
import random

import numpy as np

from visualize import visualize_frank_wolfe, generate_simplex_gif


class Simulator:
    def __init__(self, action_space: np.array):
        self.action_space = action_space

    def eval(self, i: int):
        return self.action_space[i]


class BanditsSimulator(Simulator):

    def __init__(self, action_space: np.array, mus: List[float], sigma: float):
        super(BanditsSimulator, self).__init__(action_space)
        self.mus = mus
        self.sigma = sigma

    def eval(self, i: int):
        return np.random.normal(self.mus[i], self.sigma)


class SimpleDesignSimulator(Simulator):

    def __init__(self, action_space: np.array, theta_star: float, sigma: float):
        super(SimpleDesignSimulator, self).__init__(action_space)
        self.theta_star = theta_star
        self.sigma = sigma

    def eval(self, i: int):
        z = self.action_space[i]
        theta = np.random.normal(self.theta_star, sigma)
        return z.T @ theta


class ConfidenceBound:

    def __init__(self, action_space: np.array):
        self.action_space = action_space
        self.action_counter = np.zeros(action_space.shape[0], dtype=int)

    def update_state(self, action_id: int):
        self.action_counter[action_id] += 1

    def eval(self):
        return 0


class BanditsConfidenceBound(ConfidenceBound):
    """Confidence bound from 'Fast Rates for Bandit Optimization with UCB Frank-Wolfe.'"""

    def __init__(self, action_space: np.array, deltas: np.array, sigma: float):
        super(BanditsConfidenceBound, self).__init__(action_space)
        self.deltas = deltas
        self.sigma = sigma

    def eval(self):
        t = int(np.sum(self.action_counter))  # Algorithm starts for t = 0
        nominator = 2 * sigma ** 2 * np.log((t + 1) / self.deltas[t])
        return np.sqrt(np.divide(
            nominator,
            self.action_counter.astype(np.float64),
            out=np.ones_like(self.action_counter, dtype=np.float64) * np.inf,
            where=self.action_counter != 0,
            dtype=np.float64))  # Returns inf for actions that haven't been played yet.


class DesignConfidenceBound(ConfidenceBound):
    """Confidence bound from 'Experimental Design for Linear Functionals in RKHS.'

    Can be applied only after we played all the actions in the action space.
    """

    def __init__(self, action_space: np.array, delta: float, sigma: float, lambd: float):
        super(DesignConfidenceBound, self).__init__(action_space)
        self.delta = delta
        self.sigma = sigma
        self.lambd = lambd

        self.d = action_space.shape[1]
        self.z_t_prod = np.zeros((self.d, self.d))

    def update_state(self, action_id: int):
        self.action_counter[action_id] += 1
        action = np.expand_dims(self.action_space[action_id], axis=0)
        self.z_t_prod += action.T @ action

    def eval(self):
        omega = self.z_t_prod / sigma ** 2 + self.lambd * np.identity(self.z_t_prod.shape[0])
        omega_inv = np.linalg.inv(omega)
        z_norms = np.apply_along_axis(lambda x: x @ omega_inv @ x.T, 1, self.action_space)
        return z_norms * (
                np.sqrt(2 * np.log(np.sqrt(np.linalg.det(omega))) / (self.delta * self.lambd ** (self.d / 2))) + 1)


def optimize_frank_wolfe(
        action_space: np.array,
        init_distribution: np.array,
        num_components: int,
        simulator: Simulator,
        cb: ConfidenceBound,
        warm_up_steps: int = 0,
        verbose: bool = False,
) -> np.array:
    """Performs Frank-Wolfe algorithm to maximize the objective function.
    Args:
        action_space (np.array): an array with possible actions to play along axis=0.
        init_distribution (np.array): initial distribution of actions - entries should be non-negative
            and should sum up to 1.
        num_components (int): limit on number of components to be obtained by the algorithm.
        simulator (Simulator): simulates gradient associated with playing a single action.
        cb (ConfidenceBound): confidence bound specific to the problem.
        warm_up_steps (int, optional): a number of random actions to play before the optimization starts.
        verbose (bool, optional): if True, logs optimization progress to terminal. Defaults to False.

    """
    counter = 0
    distribution = init_distribution
    distributions = [distribution.copy()]
    gradient = np.zeros_like(init_distribution)

    while counter < num_components + warm_up_steps:

        if counter < warm_up_steps:
            ucb = -1  # For verbose to work
            action_id = random.randint(0, action_space.shape[0] - 1)
        else:
            # Get upper confidence bound
            ucb = gradient + cb.eval()

            # Interact with the environment
            action_id = np.argmax(ucb)

        response = simulator.eval(action_id)
        cb.update_state(action_id)

        # Update
        action_vector = np.zeros_like(init_distribution)
        action_vector[action_id] = 1
        distribution += (action_vector - distribution) / (counter + 1)
        # Update only the gradient corresponding to the action taken
        gradient[action_id] += (response - gradient[action_id]) / (counter + 1)

        if verbose:
            print(f"Step: {counter}, Action: {action_id}, Response: {response}, Gradient: {gradient}, "
                  f"UCB: {ucb}, Distribution: {distribution}")

        distributions.append(distribution.copy())
        counter += 1

    return distributions


if __name__ == '__main__':
    steps = 20

    # Bandits example
    mus = [1, 5, 3]
    sigma = 1
    deltas = np.ones(steps)
    action_space = np.array([0, 1, 2])

    bs = BanditsSimulator(
        action_space=action_space,
        mus=mus,
        sigma=sigma)
    cb = BanditsConfidenceBound(
        action_space=action_space,
        deltas=deltas,
        sigma=sigma
    )
    distributions = optimize_frank_wolfe(
        action_space=action_space,
        init_distribution=np.ones(3) / 3,
        num_components=steps,
        simulator=bs,
        cb=cb,
        verbose=True
    )
    # visualize_frank_wolfe(distributions, action_space)
    generate_simplex_gif(distributions, path='./gif/bandits.gif')

    # Simple design example
    theta_star = np.array([1, 1, 1, 1])
    action_space = np.array([
        [1, 2, 3, 4],
        [0, 2, 3, 1],
        [4, 6, 5, 3],
    ])
    sigma = 1
    delta = 1
    lambd = 1

    bs = SimpleDesignSimulator(
        action_space=action_space,
        theta_star=theta_star,
        sigma=sigma)
    cb = DesignConfidenceBound(
        action_space=action_space,
        delta=delta,
        lambd=lambd,
        sigma=sigma
    )
    distributions = optimize_frank_wolfe(
        action_space=action_space,
        init_distribution=np.ones(3) / 3,
        num_components=steps,
        simulator=bs,
        cb=cb,
        warm_up_steps=0,
        verbose=True
    )


    def F(distribution):
        function_values = action_space @ theta_star
        return distribution @ function_values - function_values.max()


    generate_simplex_gif(distributions, path='./gif/simple_design.gif', F=F)
