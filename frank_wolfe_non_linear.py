from typing import List

import numpy as np

from visualize import generate_simplex_gif
from utils import SimpleDesignSimulator, Simulator, DesignConfidenceBound, ConfidenceBound


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

    def eval(self, vectors: np.array):
        X = self.action_space[self.actions_played]
        y = np.array(self.responses)[:, None]
        # Formula 5 from 'Experimental Design for Linear Functionals in RKHS.'
        return (vectors @ X.T @ np.linalg.inv(
            self.lambd * self.sigma ** 2 * np.identity(X.shape[0]) + X @ X.T) @ y).flatten()


class Gradient:
    def __init__(self, action_space: np.array, theta_est: Estimator, cb: ConfidenceBound, lambd: float):
        self.action_space = action_space
        self.restricted_best_action_space = self.action_space
        self.theta_est = theta_est
        self.cb = cb
        self.lambd = lambd

    def restrict_best_action_space(self):
        bound = self.cb.eval(self.action_space)
        if np.any(bound):
            preds = self.theta_est.eval(self.action_space)
            preds_ucb = preds + bound
            best_pred = np.argmax(preds_ucb)
            lb = preds[best_pred] - bound
            restriction_mask = preds_ucb >= lb
            self.restricted_best_action_space = self.action_space[restriction_mask]
            return bound, preds
        return bound, None

    def eval(self, distribution: np.array):
        return np.zeros(self.action_space.shape[1])


class GradientSimple(Gradient):

    def eval(self, distribution: np.array):
        V_eta = self.action_space.T @ np.diag(
            distribution) @ self.action_space
        V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(V_eta.shape[0]))
        star_id = np.argmax(np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, self.restricted_best_action_space))
        x_star = self.restricted_best_action_space[star_id]

        def partial(x):
            return - np.sum((x_star[:, None] @ x_star[None, :]) * (V_eta_inv @ x[:, None] @ x[None, :] @ V_eta_inv))

        return np.apply_along_axis(partial, 1, action_space)


class GradientOnlyNominator(Gradient):

    def eval(self, distribution: np.array):
        V_eta = self.action_space.T @ np.diag(
            distribution) @ self.action_space
        V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(V_eta.shape[0]))

        n, m = self.restricted_best_action_space.shape

        # Reshape the arrays to have compatible shapes for broadcasting
        # and compute differences between all possible row pairs. Choosing
        # a max over this set upperbounds max_z ||z - z^*||_V_eta_inv
        arr1_reshaped = self.restricted_best_action_space.reshape(n, 1, m)
        arr2_reshaped = self.restricted_best_action_space.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))
        star_id = np.argmax(np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs))
        diff_star = diffs[star_id]

        def partial(x):
            return - np.sum(
                (diff_star[:, None] @ diff_star[None, :]) * (V_eta_inv @ x[:, None] @ x[None, :] @ V_eta_inv))

        return np.apply_along_axis(partial, 1, action_space)


class GradientFull(Gradient):

    def __init__(self, action_space: np.array, theta_est: Estimator, cb: ConfidenceBound, variant: int = 0, eps=0.01):
        super(GradientFull, self).__init__(action_space, theta_est, cb, lambd)
        self.variant = variant
        self.eps = eps

    def eval(self, distribution: np.array):

        V_eta = self.action_space.T @ np.diag(distribution) @ self.action_space
        V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(V_eta.shape[0]))

        n, m = self.restricted_best_action_space.shape

        # Reshape the arrays to have compatible shapes for broadcasting
        # and compute differences between all possible row pairs. Choosing
        # a max over this set upperbounds max_z ||z - z^*||_V_eta_inv
        arr1_reshaped = self.restricted_best_action_space.reshape(n, 1, m)
        arr2_reshaped = self.restricted_best_action_space.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))
        star_id = np.argmax(np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs))
        diff_star = diffs[star_id]

        # Compute the lower bounds on the possible denominators.
        est = self.theta_est.eval(diffs)
        denominators = np.ones_like(est) * self.eps
        cbs = self.cb.eval(diffs)
        lcbs = (est - cbs).flatten()
        ucbs = est + cbs
        denominators[ucbs < 0] = ucbs[ucbs < 0] ** 2
        denominators[lcbs > 0] = lcbs[lcbs > 0] ** 2

        if self.variant == 0:
            star_id = np.argmax(np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs))
            diff_star = diffs[star_id]

            d = np.min(denominators)
        elif self.variant == 1:
            nominators = np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs)
            star_id = np.argmax(nominators / denominators)
            diff_star = diffs[star_id]

            d = denominators[star_id]
        else:
            raise NotImplemented

        def partial(x):
            return - np.sum(
                (diff_star[:, None] @ diff_star[None, :]) * (V_eta_inv @ x[:, None] @ x[None, :] @ V_eta_inv))

        gradient = np.apply_along_axis(partial, 1, action_space)
        return gradient / d


def optimize_frank_wolfe(
        init_distribution: np.array,
        num_components: int,
        simulator: Simulator,
        cb: ConfidenceBound,
        ge: Gradient,
        theta_est: Estimator,
        verbose: bool = False,
) -> np.array:
    """Performs Frank-Wolfe algorithm to maximize the objective function.
    Args:
        init_distribution (np.array): initial distribution of actions - entries should be non-negative
            and should sum up to 1.
        num_components (int): limit on number of components to be obtained by the algorithm.
        simulator (Simulator): simulates gradient associated with playing a single action.
        cb (ConfidenceBound): confidence bound specific to the problem.
        theta_est: (Estimator): estimator of the action_spce @ theta_est
        verbose (bool, optional): if True, logs optimization progress to terminal. Defaults to False.

    """
    counter = 0
    distribution = init_distribution

    # For visualization purpose
    distributions = [distribution.copy()]
    restricted_best_action_space = [ge.restricted_best_action_space.copy()]

    gradient = np.zeros_like(init_distribution)

    while counter < num_components:

        # Select action to play with random tie breaking
        action_id = np.random.choice(np.flatnonzero(gradient == gradient.min()))

        # Interact with the environment
        response = simulator.eval(action_id)
        cb.update_state(action_id)
        theta_est.update(action_id, response)

        # Update
        action_vector = np.zeros_like(init_distribution)
        action_vector[action_id] = 1
        distribution += (action_vector - distribution) / (counter + 1)

        # Restrict action space based on UCB
        bound, preds = ge.restrict_best_action_space()

        restricted_best_action_space.append(ge.restricted_best_action_space.copy())

        # Compute gradient for the next step
        gradient = ge.eval(distribution)

        if verbose:
            print(f"Step: {counter}, Action: {action_id}, Response: {response}, Gradient: {gradient}, "
                  f"Confidence Bound: {bound}, Distribution: {distribution},")
            if preds is not None:
                print(f"Predictions: {preds}")

        distributions.append(distribution.copy())
        counter += 1

    return distributions, restricted_best_action_space


if __name__ == '__main__':
    steps = 20

    theta_star = np.array([1, 1, 1])
    action_space = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1 / 2],
    ])
    # action_space /= 100.
    sigma = 1
    delta = 1
    # Setting lambda to 1/number of iterations we want to make.
    lambd = 1 / steps

    bs = SimpleDesignSimulator(
        action_space=action_space,
        theta_star=theta_star,
        sigma=sigma)
    # cb = ConfidenceBound(
    #     action_space=action_space,
    # )
    est = Estimator(
        action_space=action_space,
        sigma=sigma,
        lambd=lambd
    )
    cb = DesignConfidenceBound(
        action_space=action_space,
        delta=delta,
        lambd=lambd,
        sigma=sigma
    )
    # ge = GradientSimple(
    #     action_space=action_space,
    #     theta_est=est,
    #     cb=cb,
    #     lambd=lambd
    # )
    # ge = GradientOnlyNominator(
    #     action_space=action_space,
    #     theta_est=est,
    #     cb=cb,
    #     lambd=lambd
    # )
    ge = GradientFull(
        action_space=action_space,
        theta_est=est,
        cb=cb,
        variant=1
    )
    distributions, restricted_action_spaces = optimize_frank_wolfe(
        init_distribution=np.ones(3) / 3,
        num_components=steps,
        simulator=bs,
        cb=cb,
        ge=ge,
        theta_est=est,
        verbose=True
    )


    def F_simple(distribution, restricted_action_space, lambd):
        V_eta = action_space.T @ np.diag(distribution) @ action_space
        V_eta_inv = np.linalg.inv(V_eta + lambd * np.identity(V_eta.shape[0]))

        return np.max(np.apply_along_axis(lambda x: x @ V_eta_inv @ x[:, None], 1, restricted_action_space))


    def F_only_nominator(distribution, action_space, theta_star, lamd):
        x_star = action_space[np.argmax(action_space @ theta_star)]
        V_eta = action_space.T @ np.diag(distribution) @ action_space
        V_eta_inv = np.linalg.inv(V_eta + lambd * np.identity(V_eta.shape[0]))
        return np.max(np.apply_along_axis(lambda x: (x - x_star) @ V_eta_inv @ (x - x_star)[:, None], 1, action_space))


    def F_full(distribution, action_space, theta_star, lambd):
        eps = 1e-08
        x_star = action_space[np.argmax(action_space @ theta_star)]
        remaining_action_space = action_space[np.any(action_space != x_star, axis=1)]
        V_eta = action_space.T @ np.diag(distribution) @ action_space
        V_eta_inv = np.linalg.inv(V_eta + lambd * np.identity(V_eta.shape[0]))

        return np.max(np.apply_along_axis(
            lambda x: (x - x_star) @ V_eta_inv @ (x - x_star)[:, None] / ((x - x_star + eps) @ theta_star) ** 2,
            1,
            remaining_action_space))


    # generate_simplex_gif(
    #     distributions,
    #     path='./gif/non_linear_design.gif',
    #     F=lambda d: F_simple(d, action_space, lambd))
    # generate_simplex_gif(
    #     distributions,
    #     path='./gif/non_linear_design.gif',
    #     F=lambda d: F_only_nominator(d, action_space, theta_star, lambd))
    generate_simplex_gif(
        distributions,
        path='./gif/non_linear_design.gif',
        F=lambda d: F_full(d, action_space, theta_star, lambd))
