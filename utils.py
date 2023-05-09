import numpy as np


class Simulator:
    def __init__(self, action_space: np.array):
        self.action_space = action_space

    def eval(self, i: int):
        return self.action_space[i]


class SimpleDesignSimulator(Simulator):

    def __init__(self, action_space: np.array, theta_star: np.array, sigma: float):
        super(SimpleDesignSimulator, self).__init__(action_space)
        self.theta_star = theta_star
        self.sigma = sigma

    def eval(self, i: int):
        z = self.action_space[i]
        theta = np.random.normal(self.theta_star, self.sigma)
        return z.T @ theta


class ConfidenceBound:

    def __init__(self, action_space: np.array):
        self.action_space = action_space
        self.action_counter = np.zeros(action_space.shape[0], dtype=int)

    def update_state(self, action_id: int):
        self.action_counter[action_id] += 1

    def eval(self):
        return 0


class DesignConfidenceBound(ConfidenceBound):
    """Confidence bound from 'Experimental Design for Linear Functionals in RKHS.'"""

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

    def eval(self, vectors: np.array):
        omega = self.z_t_prod / self.sigma ** 2
        omega_inv = np.linalg.inv(omega + self.lambd * np.identity(self.z_t_prod.shape[0]))
        z_norms = np.apply_along_axis(lambda x: x @ omega_inv @ x.T, 1, vectors)
        return z_norms * (
                np.sqrt(2 * np.log(np.sqrt(np.linalg.det(omega))) / (self.delta * self.lambd ** (self.d / 2))) + 1)
