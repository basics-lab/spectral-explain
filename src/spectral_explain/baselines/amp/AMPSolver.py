import numpy as np
from numba import float64, int64, boolean, float32
from numba.experimental import jitclass
import numba
spec = [
    ('A', float64[:, ::1]),
    ('A2', float64[:, ::1]),
    ('y', float64[:]),
    ('M', int64),
    ('N', int64),
    ('z', float64[:]),
    ('V', float64[:]),
    ('R', float64[:]),
    ('T', float64[:]),
    ('r', float64[:]),
    ('chi', float64[:]),
    ('l', float64),
    ('d', float64)
]

@numba.jit(nopython=True)
def update_damping(old_x, new_x, damping_coefficient):
    return damping_coefficient * new_x + (1.0 - damping_coefficient) * old_x


@jitclass(spec)
class AMPSolver(object):
    """ approximate message passing solver for the Standard Linear Model (SLM) """

    def __init__(self, A, y, regularization_strength, dumping_coefficient):
        """constructor

        Args:
            A: observation matrix of shape (M, N)
            y: observed value of shape (M, )
            regularization_strength: regularization parameter
            dumping_coefficient: dumping coefficient
        """
        self.A = np.ascontiguousarray(A)
        self.A2 = np.ascontiguousarray(self.A * self.A)
        self.y = np.ascontiguousarray(y)
        self.M, self.N = A.shape

        self.z = np.random.normal(0.0, 1.0, self.M)
        self.V = np.random.uniform(0.5, 1.0, self.M)
        self.R = np.random.normal(0.0, 1.0, self.N)
        self.T = np.random.uniform(0.5, 1.0, self.N)
        self.r = np.ascontiguousarray(np.zeros(self.N))  # estimator
        self.chi = np.ascontiguousarray(np.ones(self.N))  # variance
        self.l = regularization_strength  # regularization parameter
        self.d = dumping_coefficient  # dumping coefficient

    def solve(self, max_iteration=300, tolerance=1e-5, message=False):
        """AMP solver

        Args:
            max_iteration: maximum number of iterations to be used
            tolerance: stopping criterion
            message: convergence info

        Returns:
            estimated signal
        """
        convergence_flag = False
        for iteration_index in range(max_iteration):
            self.V, self.z = self.__update_V(), self.__update_z()
            self.R, self.T = self.__update_R(), self.__update_T()
            new_r, new_chi = self.__update_r(), self.__update_chi()
            old_r = self.r.copy()
            self.r = update_damping(self.r, new_r, self.d)
            self.chi = update_damping(self.chi, new_chi, self.d)
            abs_diff = np.linalg.norm(old_r - self.r) / np.sqrt(self.N)
            if abs_diff < tolerance:
                convergence_flag = True
                if message:
                    print("requirement satisfied")
                    print("abs_diff: ", abs_diff)
                    print("abs_estimate: ", np.linalg.norm(self.r))
                    print("iteration number = ", iteration_index + 1)
                    print()
                break
        if convergence_flag:
            pass
        else:
            print("Did not coverge.")
            print("abs_diff=", abs_diff)
            print("estimate norm=", np.linalg.norm(self.r))
            if np.linalg.norm(self.r) != 0.0:
                print("relative diff= ", abs_diff / np.linalg.norm(self.r))
            print("iteration num=", iteration_index + 1)
            print()

    def __update_V(self):
        """ update V

        Returns:
            new V
        """
        return self.A2 @ self.chi

    def __update_z(self):
        """ update z

        Returns:
            new z
        """
        return self.y - self.A @ self.r + (self.V / (1.0 + self.V)) * self.z


    def __update_R(self):
        """ update R

        Returns:
            new R
        """
        v1 = self.A.T @ (self.z / (1.0 + self.V))
        v2 = self.A2.T @ (1.0 / (1.0 + self.V))
        return self.r + v1 / v2

    def __update_T(self):
        """ update T

        Returns:
            new T
        """
        v = self.A2.T @ (1.0 / (1.0 + self.V))
        return 1.0 / v

    def __update_r(self):
        """ update r

        Returns:
            new r
        """
        lT = self.l * self.T
        return (self.R - lT * np.sign(self.R)) * (np.abs(self.R) > lT)

    def __update_chi(self):
        """ update chi

        Returns:
            new chi
        """
        return self.T * (np.abs(self.R) > self.l * self.T)
