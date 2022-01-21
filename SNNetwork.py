import numpy as np
from Neuron import BiasNeuron, RBFInputNeuron, RawInputNeuron, SpikingNeuron
from sklearn import utils
import pickle

threshold = 2.
tau = 7.
eta = 0.01
e = np.exp(1)

coding_interval = 25.
time_step = 0.1


# Class to construct the spiking neural network given an array whose length sets the number of layers,
# and the values are the number of neurons on each layer, without considering the input layer.
class SNNetwork:
    def __init__(self, *, network_layout: np.ndarray, terminals: int, inhib_neurons_per_layer: int or np.ndarray,
                 input_dimension: int, bias_neurons: int, neurons_per_input_component: int or np.ndarray,
                 firing_threshold: float = threshold, decay_time: float = tau, print_info: bool = False) -> None:
        self.show = print_info
        self.distinct_ys = np.ndarray
        self.distinct_labels = np.ndarray
        global tau, threshold
        tau = decay_time
        threshold = firing_threshold
        # Make sure architecture details are in np.ndarrays.
        if isinstance(neurons_per_input_component, int):
            neurons_per_input_component = np.ones(input_dimension, dtype=int) * neurons_per_input_component
        if isinstance(inhib_neurons_per_layer, int):
            inhib_neurons_per_layer = np.ones(network_layout.shape[0], dtype=int) * inhib_neurons_per_layer
            # Output layer doesn't have inhibitory neurons.
            inhib_neurons_per_layer[-1] = 0

        assert network_layout.shape[0] == inhib_neurons_per_layer.shape[0]

        # Create hidden and output layer SpikingNeurons.
        self.layers = list()
        for l in range(network_layout.shape[0]):
            neurons = network_layout[l]
            num_inhib_neurons = inhib_neurons_per_layer[l]
            self.layers.append(np.empty(neurons, dtype=object))

            if l == 0:
                connections = int(np.sum(neurons_per_input_component)) + bias_neurons
            else:
                connections = network_layout[l - 1]

            for n in range(neurons):
                if l != network_layout.shape[0]:
                    if num_inhib_neurons > 0:
                        type = -1
                        num_inhib_neurons -= 1
                    else:
                        type = 1
                    self.layers[l][n] = SpikingNeuron(layer=l, index=n, connections=connections,
                                                      number_of_terminals=terminals, type=type, tau=tau, th=threshold)
                else:
                    self.layers[l][n] = SpikingNeuron(layer=l, index=n, connections=connections,
                                                      number_of_terminals=terminals, type=1, tau=tau, th=threshold)

        # Create input layer InputNeurons that encode real-valued data using gaussian receptive fields.
        # There are neurons_per_input_component[component] InputNeurons per input component with overlapping
        # receptive fields.
        self.input_layer = list()
        for component in range(input_dimension):
            n = neurons_per_input_component[component]
            self.input_layer.append(np.empty(n, dtype=object))
            if n == 1:  # if only 1 neuron, the input component is encoded directly into a fire time (not recommended)
                self.input_layer[component][0] = RawInputNeuron()
            else:  # if more than 1 neuron, the input component is encoded with overlaping gaussian receptive fields
                sigma = 1 / (n + 1.00001)  # the extra .00001 makes sure np.arange returns n neurons
                # Edge receptive fields are 0.5 sigma away from borders (input data must be normalised to [0, 1]).
                centers = np.arange(0.5 * sigma, 1 - 0.5 * sigma, sigma)
                for neuron in range(n):
                    self.input_layer[component][neuron] = RBFInputNeuron(center=centers[neuron], sigma=sigma)
        self.bias_neurons = bias_neurons
        if bias_neurons > 0:
            self.input_layer.append(np.array([BiasNeuron()] * bias_neurons, dtype=object))
        self.history = dict()

    # Returns a list of arrays (shape consistent with network architecture) of fire times of all neurons
    def get_fire_times_of_network(self) -> list:
        fire_times = list()
        fire_times.append(self.get_input_fire_times(self.input_layer))
        for layer in self.layers:
            fire_times.append(self.get_fire_times_of_layer(layer))
        return fire_times

    # Returns the last firing times of a layer of neurons.
    @classmethod
    def get_fire_times_of_layer(cls, layer: np.ndarray) -> np.ndarray:
        preSNFtimes = list()
        for neuron in layer:
            preSNFtimes.append(neuron.get_last_fire_time)
        return np.array(preSNFtimes)

    # Returns the last firing times of input neurons.
    @classmethod
    def get_input_fire_times(cls, input_layer: list) -> np.ndarray:
        preSNFtimes = list()
        for component in input_layer:
            for neuron in component:
                preSNFtimes.append(neuron.get_last_fire_time)
        return np.array(preSNFtimes)

    # Returns the types of a layer of neurons.
    @classmethod
    def get_types_of_layer(cls, layer: np.ndarray) -> np.ndarray:
        preSNtypes = list()
        for neuron in layer:
            preSNtypes.append(neuron.get_neuron_type)
        return np.array(preSNtypes)

    # Deletes fire times of all neurons in the net
    def reset_spike_times_of_net(self) -> None:
        for layer in self.layers:
            self.reset_spike_times_of_layer(layer=layer)
        for component in self.input_layer[:-1]:
            self.reset_spike_times_of_layer(layer=component)

    # Deletes fire times of all neurons in a layer
    def reset_spike_times_of_layer(self, *, layer: np.ndarray) -> None:
        for neuron in layer:
            neuron.reset_spike_times()

    # Prints network properties
    def displaySNN(self) -> None:
        print('------------ Displaying the network properties ------------')
        for i, component in enumerate(self.input_layer[:-1]):
            receptive_fields = component.shape[0]
            print('Input component ', i, ' has ', receptive_fields, ' encoding neurons')
            for j, neuron in enumerate(component):
                print('Encoding neuron ', j, ' of input component ', i, ' has the following properties:')
                neuron.displaySN()
        print('Input component ', len(self.input_layer) - 1, 'is a bias neuron')
        for idx in range(len(self.layers)):
            self.display_layer(layer_idx=idx)

    # Prints layer properties
    def display_layer(self, *, layer_idx) -> None:
        layer = self.layers[layer_idx]
        neurons = layer.shape[0]
        print('Layer ', layer_idx, ' has ', neurons, ' neurons.')
        for j, neuron in enumerate(layer):
            print('Neuron ', j, ' has the following properties:')
            neuron.displaySN()

    # Fits the network to x_data minimizing the loss function between network outputs and y_data
    def fit(self, *, x_data: np.ndarray, y_data: np.ndarray, y_labels: np.ndarray, learning_rate: float, epochs: int,
            val_split: float = 0.2, shuffle: bool = True, show_detailed: bool = True, path: str = '') -> None:
        global eta
        eta = learning_rate
        self.distinct_ys = np.unique(y_data, axis=0)
        self.distinct_labels = np.unique(y_labels, axis=0)
        n = x_data.shape[0]

        # shuffle data
        if shuffle:
            x_data, y_data, y_labels = utils.shuffle(x_data, y_data, y_labels)

        # split the data into training and validation sets
        split_idx = int((1 - val_split) * n)
        train_x = x_data[:split_idx]
        train_y = y_data[:split_idx]
        train_l = y_labels[:split_idx]
        val_x = x_data[split_idx:]
        val_y = y_data[split_idx:]
        val_l = y_labels[split_idx:]
        quarter = split_idx // 4 + 1
        history = {'training loss': list(), 'validation loss': list(), 'training accuracy': list(),
                   'validation accuracy': list()}

        # execute the training
        for epoch in range(epochs):
            epoch_loss, epoch_acc = 0, 0
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            for i, x_data_instance in enumerate(train_x):
                y_data_instance = train_y[i]
                y_label_instance = train_l[i]

                # predict output and label of x_data_instance
                ta = self.predict(x_data_instance=x_data_instance)  # output
                predicted_label = self.classify(output=ta)  # label
                epoch_loss = (i * epoch_loss + self.loss(ta, y_data_instance)) / (i + 1)
                if predicted_label == y_label_instance:
                    epoch_acc += 1

                # update weights
                self.backward_prop(label=y_data_instance)

                # print training summary after every 25% of training set
                if show_detailed and i % quarter == 0:
                    p = round(100 * i / split_idx, 0)
                    p10 = int(p / 5)
                    print('{}/{} '.format(i + 1, train_x.shape[0]) +
                          '[{}{}] - training loss: {}'.format(p10 * '=', (20 - p10) * ' ', round(epoch_loss, 3)),
                          ' - acc: {}%'.format(round(100 * epoch_acc / (i + 1), 1)))
            epoch_acc /= split_idx

            # save weights after every epoch for inspection (debugging / study of network learning dynamics)
            if path != '':
                pickle.dump(self, open(path + '/weights_e{}.pkl'.format(epoch + 1), 'wb'))

            # validate
            val_loss, val_acc = self.validate(val_x, val_y, val_l)

            # store weights with lowest validation loss separately, so they can be restored
            if epoch == 0:
                self.update_best()
            elif val_loss < np.amin(history['validation loss']):
                self.update_best()

            # print training summary for this epoch
            print('{}/{} [====================]'.format(train_x.shape[0], train_x.shape[0]) +
                  ' - train loss: {} - val loss: {} - train acc: {}% - val acc: {}%'.format(
                      round(epoch_loss, 3), round(val_loss, 3), round(100 * epoch_acc, 1), round(100 * val_acc, 1)))

            # update model history
            history['training loss'].append(epoch_loss)
            history['validation loss'].append(val_loss)
            history['training accuracy'].append(epoch_acc)
            history['validation accuracy'].append(val_acc)

            # naive early stopping mechanism -> if 5 period MA rises by 5% in relation to 10 period MA, stop learning
            if epoch > 10:
                if np.average(history['training loss'][-5:]) > 1.05 * np.average(history['training loss'][-10:]):
                    print('Loss stopped decreasing')
                    break
        # remap history lists to numpy arrays
        history['training loss'] = np.array(history['training loss'])
        history['validation loss'] = np.array(history['validation loss'])
        history['training accuracy'] = np.array(history['training accuracy'])
        history['validation accuracy'] = np.array(history['validation accuracy'])
        self.history = history

    # Propagate input through network and return the output in spike times
    def predict(self, *, x_data_instance: np.ndarray) -> np.ndarray or float:
        # first reset spike times from previous predictions
        self.reset_spike_times_of_net()
        # propagate the input through the network
        self.forward_prop(x_data_instance)
        # the prediction are the spike times of the output (last) layer
        ta = self.get_fire_times_of_layer(layer=self.layers[-1])
        return ta

    # Transform output spike times into a label according to closest match
    # alternatively one can match output to labels via first firing time of the output layer
    def classify(self, *, output: np.ndarray or float) -> int:
        d_min = np.inf
        output_label = 0
        for td, l in zip(self.distinct_ys, self.distinct_labels):
            d = np.linalg.norm(output - td)
            if d < d_min:
                d_min = d
                output_label = l
        return output_label

    # Function to propagate validation inputs and return metrics
    def validate(self, x_data: np.ndarray, y_data: np.ndarray, y_labels: np.ndarray) -> (float, float):
        global eta
        loss = 0
        acc = 0
        for i, x_data_instance in enumerate(x_data):
            y_data_instance = y_data[i]
            y_label_instance = y_labels[i]
            # predict label of x_data_instance
            ta = self.predict(x_data_instance=x_data_instance)
            loss = (i * loss + self.loss(ta, y_data_instance)) / (i + 1)
            predicted_label = self.classify(output=ta)
            if predicted_label == y_label_instance:
                acc += 1
        acc /= x_data.shape[0]
        return loss, acc

    # Simple squared error loss function
    def loss(self, ta, td):
        dt = ta - td
        return 0.5 * np.dot(dt, dt)

    # Function to simulate a forward pass through the network.
    def forward_prop(self, x_data: np.ndarray) -> None:
        self.input_layer = self.encode_input(x_data)
        preSNF_times = self.get_input_fire_times(self.input_layer)  # PreSynaptic Neuron Fire times
        if self.show:
            print('During forward propagation:')
            print('Input layer firing times:', preSNF_times)
        preSN_types = np.ones_like(preSNF_times)  # PreSynaptic Neuron types (input neurons all excitatory (+1))
        for l, layer in enumerate(self.layers):
            layer = self.forward_prop_through_layer(preSNF_times, preSN_types, layer)
            preSNF_times = self.get_fire_times_of_layer(layer)
            preSN_types = self.get_types_of_layer(layer)
            if self.show:
                print('Layer', l + 1, 'firing times:', preSNF_times)
            self.layers[l] = layer
        if self.show:
            print('')

    # Returns the input layer with input data encoded into spike times
    def encode_input(self, x_data: np.ndarray) -> list:
        assert x_data.shape[0] == len(self.input_layer) - 1  # input layer has BiasNeurons for reference
        input_layer = self.input_layer
        for i, x_i in enumerate(x_data):
            for neuron in input_layer[i]:
                neuron.encode(input=x_i)
        return input_layer

    # Returns the current layer whose spiking times have been updated after the signal has gone through the layer
    @classmethod
    def forward_prop_through_layer(cls, preSNF_times: np.ndarray, preSN_types: np.ndarray,
                                   curr_layer: np.ndarray or object) -> np.ndarray:
        # When there is only one neuron on the layer, an object is returned instead of an array, so recast it.
        if type(curr_layer) is not np.ndarray:
            tmp = [curr_layer]
            curr_layer = np.asarray(tmp)

        # Simulate the passes through the layer.
        for neuron in curr_layer:
            for t in np.arange(0, coding_interval, time_step):
                # SpikingNeuron action_pot function updates the firing times of the object instance
                # Returns True, if the neuron fired
                flag = neuron.action_pot(preSNF_times=preSNF_times, current_time=t, preSN_types=preSN_types)
                if flag:
                    break
        return curr_layer

    # Store current network weights as best_weights
    def update_best(self) -> None:
        for layer in self.layers:
            for neuron in layer:
                neuron.update_best_weights()

    # Replace network weights with previously stored best_weights
    def restore_best(self) -> None:
        for layer in self.layers:
            for neuron in layer:
                neuron.restore_best_weights()

    # Weight correction for backpropagation
    def delta_w(self, y: float, delta: float) -> float:
        global eta
        return -eta * y * delta

    # Propagate the error backwards and update weights
    def backward_prop(self, label: np.ndarray) -> None:
        # output layer delta (n-th hidden layer)
        delta = self.delta_output_layer(label)
        deltas = [delta, ]
        num_layers = len(self.layers)

        if num_layers > 1:  # the network has hidden layers

            # first calculate the deltas with the delta from the next layer (previous in terms of backpropagation)
            for l, layer in reversed(list(enumerate(self.layers[:-1]))):
                # l = 0, ..., n-2 hidden layers in reverse order

                if l != 0:  # 1, ..., n-2 hidden layers
                    delta = self.delta_hidden_layer(gamma_j_layer=self.layers[l + 1], gamma_i_layer=layer,
                                                    gamma_h_layer=self.layers[l - 1], delta_j_layer=delta)

                else:  # 0th hidden layer, whose previous layer is the input layer
                    delta = self.delta_hidden_layer(gamma_j_layer=self.layers[l + 1], gamma_i_layer=layer,
                                                    gamma_h_layer=self.input_layer, delta_j_layer=delta)
                deltas = [delta] + deltas

            # then correct weights of all neurons with calculated deltas (the order of weight correction doesn't matter)
            for l, layer in enumerate(self.layers):
                delta = deltas[l]

                if l != 0:  # 1, ..., n-1 hidden layers (n-1th 'hidden' layer is the output layer)
                    ti = self.get_fire_times_of_layer(self.layers[l - 1])

                else:  # 0th hidden layer, whose previous layer is the input layer
                    ti = self.get_input_fire_times(self.input_layer)

                for j, neuron in enumerate(layer):
                    m = neuron.m
                    dw = np.zeros((ti.shape[0], m))
                    t_j = neuron.get_last_fire_time
                    for i, t_i in enumerate(ti):
                        if t_i != -1:  # neuron returns -1 fire time if it didn't fire
                            for k in range(m):
                                dk = neuron.synapses[0].delays[k]  # all neurons and all synapses have same delays
                                dw[i, k] = self.delta_w(neuron.alpha_func(time=t_j - t_i - dk), delta[j])
                    neuron.update_weights(delta_w=dw)  # update weights

        else:  # the network only has an input and output layer
            for j, neuron in enumerate(self.layers[-1]):
                m = neuron.m
                ta_j = neuron.get_last_fire_time
                ti = self.get_input_fire_times(input_layer=self.input_layer)
                dw = np.zeros((ti.shape[0], m))
                for i, t_i in enumerate(ti):
                    if t_i != -1:  # neuron returns -1 fire time if it didn't fire
                        for k in range(m):
                            dk = neuron.synapses[0].delays[k]  # all neurons and all synapses have same delays
                            dw[i, k] = self.delta_w(neuron.alpha_func(time=ta_j - t_i - dk), delta[j])
                neuron.update_weights(delta_w=dw)  # update weights

    # Calculates the deltas (gradients) for the output layer
    def delta_output_layer(self, td: np.ndarray) -> np.ndarray:
        ta = self.get_fire_times_of_layer(layer=self.layers[-1])
        assert ta.shape == td.shape
        d = td - ta

        # if the net has hidden layers
        if len(self.layers) > 1:
            gamma_j_layer = self.layers[-1]
            gamma_i_layer = self.layers[-2]
            for j, neuron_j in enumerate(gamma_j_layer):
                ta_j = ta[j]
                if ta_j != -1:
                    s = 0
                    for i, neuron_i in enumerate(gamma_i_layer):
                        t_i = neuron_i.get_last_fire_time
                        if t_i != -1:
                            sign = neuron_i.type
                            for k in range(neuron_j.m):
                                w_ijk = neuron_j.synapses[i].weights[k]
                                d_k = neuron_j.synapses[i].delays[k]
                                t = ta_j - t_i - d_k
                                if t == 0:
                                    s += sign * w_ijk * e / tau
                                elif t > 0:
                                    s += sign * w_ijk * (1 / t - 1 / tau) * neuron_j.alpha_func(time=t)
                    d[j] /= s
                else:
                    d[j] = 0

        # if the net only has an input and output layer
        else:
            gamma_j_layer = self.layers[-1]
            gamma_i_layer = self.input_layer
            for j, neuron_j in enumerate(gamma_j_layer):
                if ta[j] != -1:
                    s = 0
                    n = 0  # counter for input layer neurons
                    m = neuron_j.m
                    for c, component in enumerate(gamma_i_layer):
                        for i, neuron_i in enumerate(component):
                            t_i = neuron_i.get_last_fire_time
                            if t_i != -1:
                                for k in range(m):
                                    w_ijk = neuron_j.synapses[n].weights[k]
                                    d_k = neuron_j.synapses[n].delays[k]
                                    t = ta[j] - t_i - d_k
                                    if t == 0:
                                        s += w_ijk * e / tau
                                    elif t > 0:
                                        s += w_ijk * (1 / t - 1 / tau) * neuron_j.alpha_func(time=t)
                            n += 1
                    d[j] /= s
                else:
                    d[j] = 0
        return d

    # Calculates the deltas (gradients) for the hidden layers
    def delta_hidden_layer(self, gamma_j_layer: np.ndarray, gamma_i_layer: np.ndarray,
                           gamma_h_layer: np.ndarray or list, delta_j_layer: np.ndarray) -> np.ndarray:
        d = np.zeros(gamma_i_layer.shape[0])
        tj = self.get_fire_times_of_layer(layer=gamma_j_layer)
        ti = self.get_fire_times_of_layer(layer=gamma_i_layer)

        # if the h layer is a hidden layer
        if isinstance(gamma_h_layer, np.ndarray):
            for i, neuron_i in enumerate(gamma_i_layer):
                s_j = 0
                s_h = 0
                t_i = ti[i]
                if t_i != -1:
                    sign_i = neuron_i.type

                    # contribution from the next layer (j) with inputs from the current layer (i)
                    for j, neuron_j in enumerate(gamma_j_layer):
                        t_j = tj[j]
                        if t_j != -1:
                            for k in range(neuron_i.m):
                                w_ijk = neuron_j.synapses[i].weights[k]
                                d_k = neuron_j.synapses[i].delays[k]
                                t = t_j - t_i - d_k
                                if t == 0:
                                    s_j += sign_i * delta_j_layer[j] * w_ijk * e / tau
                                elif t > 0:
                                    s_j += sign_i * delta_j_layer[j] * w_ijk * (
                                            1 / t - 1 / tau) * neuron_i.alpha_func(time=t)

                    # contribution from the current layer (i) with inputs from the previous layer (h)
                    for h, neuron_h in enumerate(gamma_h_layer):
                        t_h = neuron_h.get_last_fire_time
                        if t_h != -1:
                            sign_h = neuron_h.type
                            for k in range(neuron_i.m):
                                w_hik = neuron_i.synapses[h].weights[k]
                                d_k = neuron_i.synapses[h].delays[k]
                                t = t_i - t_h - d_k
                                if t == 0:
                                    s_h += sign_h * w_hik * e / tau
                                elif t > 0:
                                    s_h += sign_h * w_hik * (1 / t - 1 / tau) * neuron_i.alpha_func(time=t)

                    d[i] = s_j / s_h  # delta_i

        # if the h layer is the input layer
        else:
            for i, neuron_i in enumerate(gamma_i_layer):
                s_j = 0
                s_h = 0
                t_i = ti[i]
                if t_i != -1:
                    sign_i = neuron_i.type

                    # contribution from the previous layer (j) with inputs from the current layer (i)
                    for j, neuron_j in enumerate(gamma_j_layer):
                        t_j = tj[j]
                        if t_j != -1:
                            for k in range(neuron_i.m):
                                w_ijk = neuron_j.synapses[i].weights[k]
                                d_k = neuron_j.synapses[i].delays[k]
                                t = t_j - t_i - d_k
                                if t == 0:
                                    s_j += sign_i * delta_j_layer[j] * w_ijk * e / tau
                                elif t > 0:
                                    s_j += sign_i * delta_j_layer[j] * w_ijk * (
                                            1 / t - 1 / tau) * neuron_i.alpha_func(time=t)

                    # contribution from the current layer (i) with inputs from the input layer (h)
                    for c, component in enumerate(gamma_h_layer):
                        n = component.shape[0]
                        for h, neuron_h in enumerate(component):
                            t_h = neuron_h.get_last_fire_time
                            if t_h != -1:
                                for k in range(neuron_i.m):
                                    w_hik = neuron_i.synapses[c * n + h].weights[k]
                                    d_k = neuron_i.synapses[c * n + h].delays[k]
                                    t = t_i - t_h - d_k
                                    if t == 0:
                                        s_h += w_hik * e / tau
                                    elif t > 0:
                                        s_h += w_hik * (1 / t - 1 / tau) * neuron_i.alpha_func(time=t)

                    d[i] = s_j / s_h  # delta_i

        return d

    # saves the model to path in .pkl format
    def save_weights(self, *, path: str) -> None:
        pickle.dump(self, open(path + '/weights.pkl', 'wb'))
