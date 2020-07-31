import numpy as np
import tensorflow as tf


def tsne_loss(y_true, y_pred):
    """
    t-sne loss is just kld between the high and low d joint distributions.
    :param y_true: The high-d joint distribution.
    :param y_pred: The 2d coords to convert to the low-d joint.
    :return: The tnse_loss
    """
    low_d_joint = estimate_low_d_joint_of_neighbors(y_pred)
    tsne_loss = tf.keras.losses.kl_divergence(y_true, low_d_joint)
    return tsne_loss


def estimate_low_d_joint_of_neighbors(y):
    """
    q_{ij} =
        (1 + || y_{i} - y_{j} ||^{2})^{-1} / \sum_{k \neq l}(1 + || y_{k} - y_{l} ||^{2})^{-1}
    :param y: The low d coords.
    :return: The low d joint.
    """
    distances = -squared_pairwise_distance(y)
    inv_distances = tf.math.pow(1. - distances, -1)
    inv_distances = tf.linalg.set_diag(inv_distances, tf.zeros(tf.shape(inv_distances)[0]))
    joint_of_neighbors = inv_distances / tf.reduce_sum(inv_distances)
    return joint_of_neighbors


def high_d_conditional_to_joint(conditional):
    """
    Transform high d conditional to joint.
    :param conditional:
    :return:
    """
    return conditional + tf.transpose(conditional) / (
                2. * tf.cast(tf.shape(conditional)[0], tf.float32))


def squared_pairwise_distance(x):
    """
    || x_{i} - x_{j} ||^{2}
    :param x: The coords, high or low d
    :return: The squared pairwise distance.
    """
    x_left = tf.expand_dims(x, 0)
    x_right = tf.expand_dims(x, 1)
    difference = x_left - x_right
    squared_difference = tf.square(difference)

    return tf.reduce_sum(squared_difference, axis=-1)


def estimate_high_d_conditional_of_neighbors(x):
    """
    Estimate the high d conditional.
    :param x:
    :return:
    """
    distances = -squared_pairwise_distance(x)
    sigmas = estimate_sigmas(distances)
    sigmas_squared_times_two = 2. * tf.square(sigmas)
    distances_over_sigmas_squared_times_two = distances / sigmas_squared_times_two
    conditional_of_neighbors = tf.nn.softmax(distances_over_sigmas_squared_times_two, 1)
    conditional_of_neighbors = tf.linalg.set_diag(
        conditional_of_neighbors,
        tf.zeros(tf.shape(conditional_of_neighbors)[0], dtype=tf.float32))

    return conditional_of_neighbors


def estimate_sigmas(distances,
                    tolerance=1e-4,
                    max_iter=50000,
                    lower_bound=np.float32(1e-20),
                    upper_bound=np.float32(np.float32(2147483647.0) - np.float32(100.0)),
                    target_perplexity=20.0):
    """
    Estimate sigmas from a perplexity score.
    :param distances:
    :param tolerance:
    :param max_iter:
    :param lower_bound:
    :param upper_bound:
    :param target_perplexity:
    :return:
    """
    n = tf.shape(distances)[0]

    lower_bound_arr = tf.fill([n, 1], lower_bound)
    upper_bound_arr = tf.fill([n, 1], upper_bound)
    current_sigmas = tf.reshape(((lower_bound_arr + upper_bound_arr) / 2.), (-1, 1))
    done_mask = tf.fill([n], False)

    def cond(done_mask, *_):
        test = tf.math.reduce_all(done_mask)
        return not test

    def body(done_mask_inner, lower_bound_arr_inner, upper_bound_arr_inner, current_sigmas_inner):
        current_sigmas_inner = tf.reshape(((lower_bound_arr_inner + upper_bound_arr_inner) / 2.),
                                          (-1, 1))
        current_sigmas_squared_times_two = 2. * tf.square(current_sigmas_inner)
        distances_over_sigmas_squared_times_two = distances / current_sigmas_squared_times_two

        current_conditional_prob_of_neighbors = tf.nn.softmax(
            distances_over_sigmas_squared_times_two, axis=1)
        current_conditional_prob_of_neighbors = tf.linalg.set_diag(
            current_conditional_prob_of_neighbors,
            tf.zeros(n, dtype=tf.float32))

        current_perplexity = compute_perplexity(current_conditional_prob_of_neighbors)

        upper_mask = current_perplexity > target_perplexity
        upper_indices = tf.reshape(tf.where(upper_mask), (-1, 1))
        lower_mask = current_perplexity < target_perplexity
        lower_indices = tf.reshape(tf.where(lower_mask), (-1, 1))

        done_mask_inner = tf.abs(current_perplexity - target_perplexity) <= tolerance
        upper_bound_arr_inner = tf.tensor_scatter_nd_update(upper_bound_arr_inner, upper_indices,
                                                            current_sigmas_inner[upper_mask])
        lower_bound_arr_inner = tf.tensor_scatter_nd_update(lower_bound_arr_inner, lower_indices,
                                                            current_sigmas_inner[lower_mask])

        return done_mask_inner, lower_bound_arr_inner, upper_bound_arr_inner, current_sigmas_inner

    done_mask, lower_bound_arr, upper_bound_arr, current_sigmas_out = tf.while_loop(
        cond, body, [done_mask, lower_bound_arr, upper_bound_arr, current_sigmas])

    return current_sigmas_out


def log2(x):
    """
    Log2 isn't just in tf??
    :param x: The matrix to take log2 of.
    :return: log2 of a matrix.
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def compute_perplexity(distribution):
    """
    Computes the perplexity of a distribution.
    :param distribution: The distribution to compute the perplexity of.
    :return: The perplexity.
    """
    distribution += 1e-8
    entropy = -tf.reduce_sum(distribution * log2(distribution), 1)
    perplexity = 2. ** entropy
    return perplexity
