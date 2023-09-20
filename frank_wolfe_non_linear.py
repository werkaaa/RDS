from typing import List

import numpy as np

from visualize import generate_simplex_gif
from utils import Estimator, SimpleDesignSimulator, Simulator, DesignConfidenceBound, ConfidenceBound


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
        self.denominator = denominators

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
    denominators = [np.ones(action_space.shape[0] ** 2) * 0.01]

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
        if hasattr(ge, "denominator"):
            denominators.append(ge.denominator.copy())

        if verbose:
            print(f"Step: {counter}, Action: {action_id}, Response: {response}, Gradient: {gradient}, "
                  f"Confidence Bound: {bound}, Distribution: {distribution},")
            if preds is not None:
                print(f"Predictions: {preds}")

        distributions.append(distribution.copy())
        counter += 1

    return distributions, restricted_best_action_space, denominators


if __name__ == '__main__':
    steps = 30
    np.random.seed(0)

    theta_star = np.array([1, 1, 1])
    action_space = np.array([
        [0.5, 0, 0],
        [0, 1., 0],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    ])
    # action_space /= 100.
    sigma = 1
    delta = 0.1
    # Setting lambda to 1/number of iterations we want to make.
    lambd = 1 / 20

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
    ge = GradientSimple(
        action_space=action_space,
        theta_est=est,
        cb=cb,
        lambd=lambd
    )
    # ge = GradientOnlyNominator(
    #     action_space=action_space,
    #     theta_est=est,
    #     cb=cb,
    #     lambd=lambd
    # )
    # ge = GradientFull(
    #     action_space=action_space,
    #     theta_est=est,
    #     cb=cb,
    #     variant=1
    # )
    distributions, restricted_action_spaces, all_denominators = optimize_frank_wolfe(
        # init_distribution=np.ones(3) / 3,
        init_distribution=np.array([0.5, 0.5, 0]),
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


    def F_only_nominator(distribution, restricted_action_space, theta_star, lambd):
        x_star = action_space[np.argmax(action_space @ theta_star)]
        V_eta = action_space.T @ np.diag(distribution) @ action_space
        V_eta_inv = np.linalg.inv(V_eta + lambd * np.identity(V_eta.shape[0]))
        return np.max(
            np.apply_along_axis(lambda x: (x - x_star) @ V_eta_inv @ (x - x_star)[:, None], 1, restricted_action_space))


    def F_only_nominator_approx(distribution, restricted_action_space, lambd):
        V_eta = action_space.T @ np.diag(
            distribution) @ action_space
        V_eta_inv = np.linalg.inv(V_eta + lambd * np.identity(V_eta.shape[0]))

        n, m = restricted_action_space.shape

        # Reshape the arrays to have compatible shapes for broadcasting
        # and compute differences between all possible row pairs. Choosing
        # a max over this set upperbounds max_z ||z - z^*||_V_eta_inv
        arr1_reshaped = restricted_action_space.reshape(n, 1, m)
        arr2_reshaped = restricted_action_space.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))
        star_id = np.argmax(np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs))
        diff_star = diffs[star_id]
        return np.max(
            np.apply_along_axis(lambda x: diff_star @ V_eta_inv @ diff_star[:, None], 1, restricted_action_space))


    def F_full(distribution, restricted_action_space, theta_star, lambd):
        eps = 1e-08
        x_star = action_space[np.argmax(action_space @ theta_star)]
        remaining_action_space = restricted_action_space[np.any(restricted_action_space != x_star, axis=1)]
        V_eta = action_space.T @ np.diag(distribution) @ action_space
        V_eta_inv = np.linalg.inv(V_eta + lambd * np.identity(V_eta.shape[0]))

        return np.max(np.apply_along_axis(
            lambda x: (x - x_star) @ V_eta_inv @ (x - x_star)[:, None] / ((x - x_star + eps) @ theta_star) ** 2,
            1,
            restricted_action_space))


    def F_full_approx(distribution, restricted_action_space, denominators, lambd):
        V_eta = action_space.T @ np.diag(
            distribution) @ action_space
        V_eta_inv = np.linalg.inv(V_eta + lambd * np.identity(V_eta.shape[0]))

        n, m = restricted_action_space.shape

        # Reshape the arrays to have compatible shapes for broadcasting
        # and compute differences between all possible row pairs. Choosing
        # a max over this set upperbounds max_z ||z - z^*||_V_eta_inv
        arr1_reshaped = restricted_action_space.reshape(n, 1, m)
        arr2_reshaped = restricted_action_space.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))
        nominators = np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs)
        return np.max(nominators / denominators) / 20.


    generate_simplex_gif(
        distributions,
        path='./gif/non_linear_design.gif',
        F1=lambda d: F_simple(d, action_space, lambd),
        F2=[lambda d, ras=restricted_action_space: F_simple(d, ras, lambd) for restricted_action_space in
            restricted_action_spaces])
    # generate_simplex_gif(
    #     distributions,
    #     path='./gif/non_linear_design_nominator.gif',
    #     F2=[lambda d, ras=restricted_action_space: F_only_nominator_approx(d, ras, lambd) for restricted_action_space in
    #         restricted_action_spaces],
    #     F1=[lambda d, ras=restricted_action_space: F_only_nominator(d, ras, theta_star, lambd) for
    #         restricted_action_space in
    #         restricted_action_spaces])
    # generate_simplex_gif(
    #     distributions,
    #     path='./gif/non_linear_design_full.gif',
    #     F1=[lambda d, ras=restricted_action_space: F_full(d, ras, theta_star, lambd) for restricted_action_space in
    #         restricted_action_spaces],
    #     F2=[lambda d, ras=restricted_action_space, den=denominators: F_full_approx(d, ras, den, lambd) for
    #         (restricted_action_space, denominators) in
    #         zip(restricted_action_spaces, all_denominators)])
