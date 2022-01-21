import numpy as np
from SNNetwork import SNNetwork
from sklearn.datasets import load_iris
import pickle
import os
import time

# ----------------------------------------------------------------------------------------------


def check_n_create_path(path):
    i = 1
    n = len(path)
    while os.path.isdir(path):
        if i == 1:
            path += str(i)
        else:
            path = path[:n] + str(i)
        i += 1
    os.mkdir(path)
    return path


def generate_xor_data(n, sigma):
    n_half = n // 2
    x = np.zeros((n, 2))
    x[:n_half, 0] = np.random.normal(0.25, sigma, n_half)
    x[:n_half, 1] = np.random.normal(0.25, sigma, n_half)
    x[x < 0] = -x[x < 0]
    x[n_half:, 0] = np.random.normal(0.75, sigma, n - n_half)
    x[n_half:, 1] = np.random.normal(0.75, sigma, n - n_half)
    x[x > 1] = -x[x > 1] % 1
    y = np.zeros(n)
    y[n_half:] = 1
    return x, y


def generate_xor_moon(n, sigma):
    n_half = n // 2
    x = np.zeros((n, 2))
    x[:n_half, 0] = np.linspace(0.1, 0.7, n_half) + np.random.normal(0, sigma, n_half)
    x[:n_half, 1] = 0.5 + 0.4 * np.sin(np.linspace(0, np.pi, n_half)) + np.random.normal(0, sigma, n_half)
    x[x < 0] = -x[x < 0]
    x[n_half:, 0] = np.linspace(0.3, 0.9, n_half) + np.random.normal(0, sigma, n_half)
    x[n_half:, 1] = 0.5 - 0.4 * np.sin(np.linspace(0, np.pi, n_half)) + np.random.normal(0, sigma, n - n_half)
    x[x > 1] = -x[x > 1] % 1
    y = np.zeros(n)
    y[n_half:] = 1
    return x, y


def main_iris():
    output_dim = 3
    hidden_layer_num = 10
    terminal_num = 8
    input_encoder_num = 8
    inhib_num = 2
    model_path = 'Models/iris-{}-{}-{}({})'.format(input_encoder_num, hidden_layer_num, output_dim, terminal_num)
    # model_path = 'Models/iris-{}-{}({})'.format(input_encoder_num, output_dim, terminal_num)
    model_path = check_n_create_path(model_path)
    iris = load_iris()
    x_data = iris.data
    y_labels = iris.target
    y_times = np.zeros((y_labels.shape[0], output_dim))
    for i, y in enumerate(y_labels):
        if y == 0:
            y_times[i] = [10, 16, 16]
        elif y == 1:
            y_times[i] = [16, 10, 16]
        elif y == 2:
            y_times[i] = [16, 16, 10]
    x_data = (x_data - np.amin(x_data)) / (np.amax(x_data) - np.amin(x_data))

    net = SNNetwork(network_layout=np.array([hidden_layer_num, output_dim, ]), terminals=terminal_num,
                    inhib_neurons_per_layer=inhib_num, input_dimension=x_data.shape[1], bias_neurons=1,
                    neurons_per_input_component=input_encoder_num, firing_threshold=2., decay_time=7.)

    t0 = time.time()
    net.fit(x_data=x_data, y_data=y_times, y_labels=y_labels, learning_rate=0.01, epochs=10, show_detailed=True,
            path=model_path, val_split=0.5)
    np.save(model_path + '/t', t0 - time.time())
    net.save_weights(path=model_path)

    net = pickle.load(open(model_path + '/weights.pkl', 'rb'))
    net.show = True
    for i, x in enumerate(x_data[::50]):
        a = net.predict(x_data_instance=x)
        print(x, a, y_times[::50][i])


def main_xor():
    output_dim = 2
    hidden_layer_num = 5
    terminal_num = 8
    input_encoder_num = 4
    inhib_num = 1
    model_path = 'Models/xor-{}-{}-{}({})'.format(input_encoder_num, hidden_layer_num, output_dim, terminal_num)
    # model_path = 'Models/xor-{}-{}({})'.format(input_encoder_num, output_dim, terminal_num)
    model_path = check_n_create_path(model_path)
    data_size = 1000
    x_data, y_labels = generate_xor_data(data_size, 0.15)

    y_times = np.zeros((y_labels.shape[0], output_dim))
    for i, y in enumerate(y_labels):
        if y == 0:
            y_times[i] = [6, 10]
        elif y == 1:
            y_times[i] = [10, 6]
    x_data = (x_data - np.amin(x_data)) / (np.amax(x_data) - np.amin(x_data))

    net = SNNetwork(network_layout=np.array([hidden_layer_num, output_dim, ]), terminals=terminal_num,
                    inhib_neurons_per_layer=inhib_num, input_dimension=x_data.shape[1], bias_neurons=1,
                    neurons_per_input_component=input_encoder_num, firing_threshold=2., decay_time=7.)

    t0 = time.time()
    net.fit(x_data=x_data, y_data=y_times, y_labels=y_labels, learning_rate=0.01, epochs=10, show_detailed=True,
            path=model_path)
    np.save(model_path + '/t', t0 - time.time())
    net.save_weights(path=model_path)

    net = pickle.load(open(model_path + '/weights.pkl', 'rb'))
    net.show = True
    for i, x in enumerate(x_data[::50]):
        a = net.predict(x_data_instance=x)
        print(x, a, y_times[::50][i])


def main_xor_moon():
    output_dim = 2
    hidden_layer_num = 5
    terminal_num = 8
    input_encoder_num = 3
    inhib_num = 1
    model_path = 'Models/xor_moon-{}-{}-{}({})'.format(input_encoder_num, hidden_layer_num, output_dim, terminal_num)
    # model_path = 'Models/xor_moon-{}-{}({})'.format(input_encoder_num, output_dim, terminal_num)
    model_path = check_n_create_path(model_path)
    data_size = 1000
    x_data, y_labels = generate_xor_moon(data_size, 0.075)

    y_times = np.zeros((y_labels.shape[0], output_dim))
    for i, y in enumerate(y_labels):
        if y == 0:
            y_times[i] = [6, 10]
        elif y == 1:
            y_times[i] = [10, 6]
    x_data = (x_data - np.amin(x_data)) / (np.amax(x_data) - np.amin(x_data))

    net = SNNetwork(network_layout=np.array([hidden_layer_num, output_dim, ]), terminals=terminal_num,
                    inhib_neurons_per_layer=inhib_num, input_dimension=x_data.shape[1], bias_neurons=1,
                    neurons_per_input_component=input_encoder_num, firing_threshold=2., decay_time=7.)

    t0 = time.time()
    net.fit(x_data=x_data, y_data=y_times, y_labels=y_labels, learning_rate=0.01, epochs=10, show_detailed=True,
            path=model_path)
    np.save(model_path + '/t', t0 - time.time())
    net.save_weights(path=model_path)

    net = pickle.load(open(model_path + '/weights.pkl', 'rb'))
    net.show = True
    for i, x in enumerate(x_data[::50]):
        a = net.predict(x_data_instance=x)
        print(x, a, y_times[::50][i])


if __name__ == "__main__":
    main_xor()
    main_xor_moon()