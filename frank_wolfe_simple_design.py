import copy
from typing import List, Optional
import random

import numpy as np

from visualize import generate_simplex_gif
from utils import Estimator, SimpleDesignSimulator, Simulator, DesignConfidenceBound, ConfidenceBound


class BanditsSimulator(Simulator):

    def __init__(self, action_space: np.array, mus: List[float], sigma: float):
        super(BanditsSimulator, self).__init__(action_space)
        self.mus = mus
        self.sigma = sigma

    def eval(self, i: int):
        return np.random.normal(self.mus[i], self.sigma)


class BanditsConfidenceBound(ConfidenceBound):
    """Confidence bound from 'Fast Rates for Bandit Optimization with UCB Frank-Wolfe.'"""

    def __init__(self, action_space: np.array, deltas: np.array, sigma: float):
        super(BanditsConfidenceBound, self).__init__(action_space)
        self.deltas = deltas
        self.sigma = sigma

    def eval(self, vectors: Optional[np.array] = None):
        t = int(np.sum(self.action_counter))  # Algorithm starts for t = 0
        nominator = 2 * sigma ** 2 * np.log((t + 1) / self.deltas[t])
        return np.sqrt(np.divide(
            nominator,
            self.action_counter.astype(np.float64),
            out=np.ones_like(self.action_counter, dtype=np.float64) * np.inf,
            where=self.action_counter != 0,
            dtype=np.float64))  # Returns inf for actions that haven't been played yet.


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
    ucbs = [None]
    gradient = np.zeros_like(init_distribution)
    theta_est = Estimator(
        action_space=action_space,
        sigma=sigma,
        lambd=lambd
    )

    while counter < num_components + warm_up_steps:

        if counter < warm_up_steps:
            ucb = -1  # For verbose to work
            ucbs.append(None)
            action_id = random.randint(0, action_space.shape[0] - 1)
        else:
            # Get upper confidence bound
            ucb = gradient + cb.eval(action_space)
            ucbs.append(ucb)

            # Interact with the environment
            action_id = np.random.choice(np.flatnonzero(ucb == ucb.max()))

        response = simulator.eval(action_id)
        theta_est.update(action_id, response)
        cb.update_state(action_id)

        # Update
        action_vector = np.zeros_like(init_distribution)
        action_vector[action_id] = 1
        distribution += (action_vector - distribution) / (counter + 1)
        gradient = theta_est.eval(action_space)

        if verbose:
            print(f"Step: {counter}, Action: {action_id}, Response: {response}, Gradient: {gradient}, "
                  f"UCB: {ucb}, Distribution: {distribution}")

        distributions.append(distribution.copy())
        counter += 1

    return distributions, ucbs


if __name__ == '__main__':
    np.random.seed(1)
    steps = 100

    # Bandits example
    mus = [4.5, 5, 4]
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
    distributions, _ = optimize_frank_wolfe(
        action_space=action_space,
        init_distribution=np.ones(3) / 3,
        num_components=steps,
        simulator=bs,
        cb=cb,
        verbose=True
    )
    # visualize_frank_wolfe(distributions, action_space)
    # generate_simplex_gif(distributions, path='./gif/bandits.gif')

    # Simple design example
    theta_star = np.array([1, 1])
    action_space = np.array([
        [1, 0],
        [0, 1],
        [np.sqrt(0.5), np.sqrt(0.5)],
    ])
    sigma = 1
    delta = 0.1
    lambd = 1 / 50

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
    distributions, ucbs = optimize_frank_wolfe(
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


    def F_ucb(distribution, ucb):
        if ucb is None:
            return 0
        return distribution @ ucb - ucb.max()


    generate_simplex_gif(distributions, path='./gif/simple_design.gif', F1=F,
                         F2=[lambda d, ucb=ucb: F_ucb(d, ucb) for ucb in ucbs])
