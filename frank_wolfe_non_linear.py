from typing import List

import numpy as np

from visualize import generate_simplex_gif
from utils import SimpleDesignSimulator, Simulator, DesignConfidenceBound, ConfidenceBound


class Gradient:
    def __init__(self, action_space: np.array):
        self.action_space = action_space
        self.restricted_action_space = self.action_space

    def restrict_action_space(self, restriction_mask: np.array):
        self.restricted_action_space = self.action_space[restriction_mask]

    def eval(self, distribution: np.array):
        V_eta = self.action_space.T @ np.diag(distribution) @ self.action_space
        if np.linalg.matrix_rank(V_eta) != V_eta.shape[0]:
            return np.zeros_like(distribution)
        V_eta_inv = np.linalg.inv(V_eta)
        star_id = np.argmax(np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, self.restricted_action_space))
        x_star = self.restricted_action_space[star_id]

        def partial(x):
            return - np.sum((x_star[:, None] @ x_star[None, :]) * (V_eta_inv @ x[:, None] @ x[None, :] @ V_eta_inv))

        return np.apply_along_axis(partial, 1, action_space)


class Estimator:
    """ Estimator of action_space @ theta_est values."""

    def __init__(self, action_space: np.array, sigma: float, lambd: float):
        self.action_space = action_space
        self.sigma = sigma
        self.lambd = lambd
        self.actions_played = []
        self.responses = []

    def update(self, action: np.array, response: float):
        self.actions_played.append(action)
        self.responses.append(response)

    def eval(self):
        X = self.action_space[self.actions_played]
        y = np.array(self.responses)[:, None]
        # Formula 5 from 'Experimental Design for Linear Functionals in RKHS.'
        return self.action_space @ X.T @ np.linalg.inv(
            self.lambd * self.sigma ** 2 * np.identity(X.shape[0]) + X @ X.T) @ y


def optimize_frank_wolfe(
        init_distribution: np.array,
        num_components: int,
        simulator: Simulator,
        cb: ConfidenceBound,
        ge: Gradient,
        est: Estimator,
        verbose: bool = False,
) -> np.array:
    """Performs Frank-Wolfe algorithm to maximize the objective function.
    Args:
        init_distribution (np.array): initial distribution of actions - entries should be non-negative
            and should sum up to 1.
        num_components (int): limit on number of components to be obtained by the algorithm.
        simulator (Simulator): simulates gradient associated with playing a single action.
        cb (ConfidenceBound): confidence bound specific to the problem.
        est: (Estimator): estimator of the action_spce @ theta_est
        verbose (bool, optional): if True, logs optimization progress to terminal. Defaults to False.

    """
    counter = 0
    distribution = init_distribution

    # For visualization purpose
    distributions = [distribution.copy()]
    restricted_action_space = [ge.restricted_action_space.copy()]

    gradient = np.zeros_like(init_distribution)

    while counter < num_components:

        # Select action to play with random tie breaking
        action_id = np.random.choice(np.flatnonzero(gradient == gradient.max()))

        # Interact with the environment
        response = simulator.eval(action_id)
        cb.update_state(action_id)
        est.update(action_id, response)

        # Update
        action_vector = np.zeros_like(init_distribution)
        action_vector[action_id] = 1
        distribution += (action_vector - distribution) / (counter + 1)

        # Restrict action space based on UCB
        bound = cb.eval()
        if np.any(bound):
            preds = est.eval().flatten()
            preds_ucb = preds + bound
            best_pred = np.argmax(preds_ucb)
            lb = preds[best_pred] - bound
            ge.restrict_action_space(preds_ucb >= lb)

        restricted_action_space.append(ge.restricted_action_space.copy())

        # Compute gradient for the next step
        gradient = ge.eval(distribution)

        if verbose:
            print(f"Step: {counter}, Action: {action_id}, Response: {response}, Gradient: {gradient}, "
                  f"Confidence Bound: {bound}, Distribution: {distribution},")
            if np.any(bound):
                print(f"Predictions: {preds}")

        distributions.append(distribution.copy())
        counter += 1

    return distributions, restricted_action_space


if __name__ == '__main__':
    steps = 20

    theta_star = np.array([1, 1])
    action_space = np.array([
        [3, 0],
        [1, 0],
        [2, 0],
    ])
    sigma = 1
    delta = 1
    lambd = 1

    bs = SimpleDesignSimulator(
        action_space=action_space,
        theta_star=theta_star,
        sigma=sigma)
    # cb = ConfidenceBound(
    #     action_space=action_space,
    # )
    cb = DesignConfidenceBound(
        action_space=action_space,
        delta=delta,
        lambd=lambd,
        sigma=sigma
    )
    ge = Gradient(
        action_space=action_space
    )
    est = Estimator(
        action_space=action_space,
        sigma=sigma,
        lambd=lambd
    )
    distributions, restricted_action_spaces = optimize_frank_wolfe(
        init_distribution=np.ones(3) / 3,
        num_components=steps,
        simulator=bs,
        cb=cb,
        ge=ge,
        est=est,
        verbose=True
    )


    def F(distribution, restricted_action_space):
        V_eta = action_space.T @ np.diag(distribution) @ action_space
        if np.linalg.matrix_rank(V_eta) != V_eta.shape[0]:
            return None
        V_eta_inv = np.linalg.inv(V_eta)
        return np.max(np.apply_along_axis(lambda x: x @ V_eta_inv @ x[:, None], 1, restricted_action_space))


    generate_simplex_gif(
        distributions,
        path='./gif/non_linear_design.gif',
        F=[lambda d: F(d, r) for r in restricted_action_spaces])
