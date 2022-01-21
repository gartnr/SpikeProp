import numpy as np


# class for initialising connections between neurons
# connections from layer i to layer i+1 are always stored as the .synapse attribute of layer i+1 neurons
class Connection:

    def __init__(self, *, number_of_terminals: int, min_delay: float, max_delay: float, tau: float, prev_layer_n: int
                 ) -> None:
        w_min = tau / (number_of_terminals * prev_layer_n * self._alpha(min_delay, tau))
        w_max = tau / (number_of_terminals * prev_layer_n * self._alpha(max_delay, tau))
        self.weights = np.random.uniform(w_min, w_max, number_of_terminals)
        self.best_weights = self.weights
        self.delays = np.linspace(min_delay, max_delay, number_of_terminals)

    def _alpha(self, t: float, tau: float) -> float:
        if t >= 0:
            div = t / tau
            return div * np.exp(1 - div)
        else:
            return 0.
