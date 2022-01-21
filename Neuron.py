import numpy as np
from Connections import Connection

# set minimum and maximum delay of terminals for the model here (only SpikingNeurons have terminals, input neurons dont)
min_delay = 1.
max_delay = 8.
decay_time = 7.


class SpikingNeuron:

    def __init__(self, *, layer: int, index: int, connections: int, number_of_terminals: int, type: int, tau: float,
                 th: float) -> None:
        global decay_time
        decay_time = tau
        self.layer = layer
        self.index = index
        self.m = number_of_terminals
        self.fire_times = list()
        self.type = type
        self.synapses = np.empty(connections, dtype=object)
        self.threshold = th
        # connection of a given neuron are connections from the previous layer neurons
        for s in range(connections):
            self.synapses[s] = Connection(number_of_terminals=number_of_terminals, min_delay=min_delay,
                                          max_delay=max_delay, tau=decay_time, prev_layer_n=connections)

    # returns the number of connections the neuron has
    @property
    def get_num_of_connections(self) -> int:
        return self.synapses.shape[0]

    # returns the last firing time of the neuron
    @property
    def get_last_fire_time(self) -> float:
        if len(self.fire_times) > 0:
            return self.fire_times[-1]
        else:
            return -1.

    # returns the type of input SpikingNeuron (1: excitatory or -1: inhibitory)
    @property
    def get_neuron_type(self) -> int:
        return self.type

    # deletes list of firing times
    def reset_spike_times(self) -> None:
        del self.fire_times[:]

    # updates neuron weights
    def update_weights(self, delta_w: np.ndarray) -> None:
        conn_num = self.get_num_of_connections
        for c in range(conn_num):
            self.synapses[c].weights = np.add(self.synapses[c].weights, delta_w[c, :])
            for i, w in enumerate(self.synapses[c].weights):
                if w < 0.:
                    self.synapses[c].weights[i] = 0.

    # updates neuron best weights
    def update_best_weights(self) -> None:
        conn_num = self.get_num_of_connections
        for c in range(conn_num):
            self.synapses[c].best_weights = self.synapses[c].weights

    # updates neuron weights to best_weights
    def restore_best_weights(self) -> None:
        conn_num = self.get_num_of_connections
        for c in range(conn_num):
            self.synapses[c].weights = self.synapses[c].best_weights

    # single input response
    def alpha_func(self, *, time: np.ndarray or float) -> np.ndarray or float:
        if isinstance(time, float):
            a = 0
            if time >= 0:
                a = time / decay_time * np.exp(1 - time / decay_time)
        else:
            a = np.zeros_like(time)
            a[time > 0] = time[time > 0] / decay_time * np.exp(1 - time[time > 0] / decay_time)
        return a

    # method to compute the internal state variable of the current neuron, in order to determine if the
    # neuron is spiking or not
    def _internal_state_variable(self, *, preSNF_times: np.ndarray, current_time: float, preSN_types: np.ndarray
                                 ) -> float:
        connections = self.get_num_of_connections
        state_variable = 0.0

        for s in range(connections):
            # if preSNF_times[s] < 0, neuron s didn't fire
            if current_time >= preSNF_times[s] >= 0:
                terminals = self.m
                for t in range(terminals):
                    state_variable += self.synapses[s].weights[t] * preSN_types[s] * self._terminal_contribution(
                        relative_time=current_time - preSNF_times[s] - self.synapses[s].delays[t])
        return state_variable

    # unweighted contribution of a terminal
    def _terminal_contribution(self, relative_time: float) -> float:
        return self.alpha_func(time=relative_time)

    # based on the internal state variable of the neuron, check if it is generating an action potential (spike) or not
    def action_pot(self, *, preSNF_times: np.ndarray, current_time: float, preSN_types: np.ndarray) -> bool:
        if len(self.fire_times) == 0:
            potential = self._internal_state_variable(preSNF_times=preSNF_times, current_time=current_time,
                                                      preSN_types=preSN_types)
            if potential >= self.threshold:
                self.fire_times.append(current_time)
                return True
            return False

    # prints the parameters and structure of a neuron
    def displaySN(self) -> None:
        print('--------------------')
        connections = self.synapses.shape
        print('The firing time of the neuron is ', self.fire_times)
        print('The type of the neuron is: ', self.type)
        print('Number of connections', connections[0])
        terminals = self.synapses[0].weights.shape
        print('Number of terminals per connection: ', terminals)
        print('Delays of connections:')
        print(self.synapses[0].delays)
        for s in range(connections[0]):
            print('Weights of connection {}: '.format(s))
            print(self.synapses[s].weights)


# Radial basis function encoding neuron (gaussian in this case)
class RBFInputNeuron:
    def __init__(self, *, center: float, sigma: float) -> None:
        self.x0 = center
        self.sig = sigma
        self.fire_times = list()
        self.input = None

    # translate real-valued input into a spike-time via gaussian receptive field
    def encode(self, *, input: float) -> None:
        self.input = input
        fire_time = max_delay * (self.gaussian(x=input) - 1) / 2
        if fire_time > max_delay:
            fire_time = -1
        self.fire_times.append(fire_time)

    # shape of receptive field
    def gaussian(self, *, x: float) -> float:
        return np.exp((x - self.x0)**2 / (2 * self.sig * self.sig))

    # returns the last firing time of the neuron
    @property
    def get_last_fire_time(self) -> float:
        if len(self.fire_times) > 0:
            return self.fire_times[-1]
        else:
            return -1

    # deletes list of firing times
    def reset_spike_times(self) -> None:
        del self.fire_times[:]

    # function to display the parameters and structure of a neuron
    def displaySN(self) -> None:
        print('--------------------')
        print('The RBF input neuron receptive field is centered at ', round(self.x0, 2), ' with a width of ',
              round(self.sig, 2))
        print('The firing time of the neuron is ', self.fire_times)


class RawInputNeuron:
    def __init__(self) -> None:
        self.fire_times = list()
        self.input = None

    # translate real-valued input into a spike-time via gaussian receptive field
    def encode(self, *, input: float) -> None:
        self.input = input
        fire_time = max_delay * input / 2
        self.fire_times.append(fire_time)

    # returns the last firing time of the neuron
    @property
    def get_last_fire_time(self) -> float:
        if len(self.fire_times) > 0:
            return self.fire_times[-1]
        else:
            return -1

    # deletes list of firing times
    def reset_spike_times(self) -> None:
        del self.fire_times[:]

    # function to display the parameters and structure of a neuron
    def displaySN(self) -> None:
        print('--------------------')
        print('The firing time of the raw input neuron is ', self.fire_times)


# Bias neuron that always fires at the same time so the network has a temporal reference point (can be set manually)
class BiasNeuron:
    def __init__(self, fire_time: float = 0.) -> None:
        self.fire_times = [fire_time]

    # returns the last firing time of the neuron
    @property
    def get_last_fire_time(self) -> float:
        return self.fire_times[-1]

    # function to display the parameters and structure of a neuron
    def displaySN(self) -> None:
        print('--------------------')
        print('The input neuron is a bias neuron')
