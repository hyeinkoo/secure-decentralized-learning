import numpy as np


def weight_initializer(weights_shape, initializer_function, **kwargs):
    """
    Will generate the weights in weight_shape, by filling each row with the given initializer function
    :param weights_shape: shape of the final weights (shape[0] are usually our data features, shape[1] are the number of
    neurons in the next layer
    :param initializer_function: a function that fills every every row in the weight matrix
    :return: initialized weights
    """
    # weight shape will be input-features (our data), output features

    row_length = weights_shape[0]
    number_rows = weights_shape[1]

    weights = np.zeros((row_length, number_rows))
    for i in range(number_rows):
        weights[:, i] = initializer_function(row_length, **kwargs)

    return weights



# we should scale down the positive weights
def generate_symmetric_points(num_elements, down_scale_factor=0.95, mu=0.0, sigma=0.5):
    """num_pos says roughly how many positive elements we want to have"""

    vector = np.random.normal(mu, sigma, num_elements)

    # make all negative (such that we can better control)
    abs_vector = abs(vector) * (-1)

    num_pos = np.floor(num_elements / 2).astype(int) # 전체 element 개수의 절반 정도를 positive로

    # random positive indices
    pos_indices = np.random.choice(num_elements, num_pos, replace=False)

    negative_elements = np.delete(abs_vector, pos_indices)

    abs_vector[pos_indices] = -down_scale_factor * negative_elements  # set the negative values and turn them positive

    return abs_vector



def adversarial_initialization(model,
                               initializer = generate_symmetric_points,
                               hyperparameter = {'down_scale_factor': 0.95},
                               features = 32*32*3,
                               num_neurons = 1000,
                               layer=1):
    weights_shape = (features, num_neurons)
    weights = weight_initializer(weights_shape, initializer, **hyperparameter)  # todo pass the kwargs
    bs = np.zeros(num_neurons)
    model.layers[layer].set_weights([weights, bs])
