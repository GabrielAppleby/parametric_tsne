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
    tsne_loss = tf.keras.losses.KLDivergence()(y_true, low_d_joint)
    return tsne_loss


def zeroed_softmax(x, idx):
    n = tf.shape(x)[0]
    scaled = tf.subtract(x, tf.reduce_max(x))

    e_x = tf.exp(scaled)
    indices = tf.concat((tf.range(idx), tf.range(idx+1, n)), axis=0)
    e_x = tf.gather(e_x, indices)
    # epsilon = tf.constant(1e-8, dtype=tf.float32)
    # epsilon = tf.add(epsilon, e_x)
    dem = tf.reduce_sum(e_x)

    test = tf.divide(e_x, dem)
    return test


def zeroed_softmax_axis(x, axis=1):
    n = tf.shape(x)[0]
    maxx = tf.reduce_max(x, axis=1, keepdims=True)
    scaled = tf.subtract(x, maxx)

    e_x = tf.exp(scaled)
    e_x = tf.linalg.set_diag(
        e_x,
        tf.zeros(tf.shape(x)[0], dtype=tf.float32))

    dem = tf.reduce_sum(e_x, axis=1, keepdims=True)

    test = tf.math.divide_no_nan(e_x, dem)
    return test


def estimate_low_d_joint_of_neighbors(y):
    """
    q_{ij} =
        (1 + || y_{i} - y_{j} ||^{2})^{-1} / \sum_{k \neq l}(1 + || y_{k} - y_{l} ||^{2})^{-1}
    :param y: The low d coords.
    :return: The low d joint.
    """
    distances = squared_pairwise_distance(y)
    inv_distances = tf.math.pow((1. + distances), -1)
    inv_distances = tf.linalg.set_diag(inv_distances, tf.zeros(tf.shape(inv_distances)[0]))
    joint_of_neighbors = inv_distances / tf.reduce_sum(inv_distances)
    return joint_of_neighbors


def high_d_conditional_to_joint(conditional):
    """
    Transform high d conditional to joint.
    :param conditional:
    :return:
    """
    joint = (conditional + tf.transpose(conditional)) / 2.0
    sumjoint = tf.cast(tf.shape(conditional[0]), tf.float32)
    normalized_joint = tf.math.divide_no_nan(joint, sumjoint)

    return normalized_joint


def squared_pairwise_distance(x):
    """
    || x_{i} - x_{j} ||^{2}
    :param x: The coords, high or low d
    :return: The squared pairwise distance.
    """
    r = tf.reduce_sum(tf.square(x), axis=1)
    r = tf.reshape(r, [-1, 1])

    squared_difference = r - 2 * tf.matmul(x, x, transpose_b=True) + tf.transpose(r)
    return squared_difference


def estimate_high_d_conditional_of_neighbors(x):
    """
    Estimate the high d conditional.
    :param x:
    :return:high_d_joints
    """
    distances = -squared_pairwise_distance(x)
    sigmas = estimate_sigmas(distances)
    sigmas_squared_times_two = tf.square(sigmas)
    distances_over_sigmas_squared_times_two = distances / sigmas_squared_times_two
    conditional_of_neighbors = zeroed_softmax_axis(distances_over_sigmas_squared_times_two, axis=1)

    return conditional_of_neighbors


# def estimate_sigmas(negative_squared_distances,
#                     tolerance=1e-4,
#                     max_iter=100,
#                     target_perplexity=30.0):
#     target_entropy = tf.math.log(target_perplexity)
#     i = tf.zeros(tf.shape(negative_squared_distances)[0], dtype=tf.int32)
#
#     def estimate_sigma(negative_squared_distance_arr, i):
#         upper_bound = tf.constant(np.float32(np.float32(2147483647.0) - np.float(100.0)))
#         lower_bound = tf.constant(1e-37)
#         current_sigma = (upper_bound + lower_bound) / 2
#         keep_going = True
#
#         def cond(keep_going, *_):
#             return keep_going
#
#         def body(keep_going, current_sigma, upper_bound, lower_bound):
#             # p = tf.exp(tf.multiply(negative_squared_distances, current_beta))
#             # print(p)
#             # sump = tf.reduce_sum(p)
#             #
#             # h = tf.math.log(sump) + current_beta * tf.reduce_sum(tf.multiply(negative_squared_distances, p)) / sump
#
#             current_sigma = (upper_bound + lower_bound) / 2
#             current_sigma_squared_times_two = 2. * tf.square(current_sigma)
#             distances_over_sigma_squared_times_two = negative_squared_distance_arr / current_sigma_squared_times_two
#             current_conditional_prob_of_neighbors = zeroed_softmax(
#                 distances_over_sigma_squared_times_two, i)
#
#             current_entropy = compute_entropy(current_conditional_prob_of_neighbors)
#
#             # print(current_sigma)
#             # print(2.0 ** current_entropy)
#
#             upper_bound = tf.cond(pred=(current_entropy > target_entropy), true_fn=lambda: current_sigma,
#                                   false_fn=lambda: upper_bound)
#             lower_bound = tf.cond(pred=(current_entropy < target_entropy), true_fn=lambda: current_sigma,
#                                   false_fn=lambda: lower_bound)
#
#             keep_going = tf.abs(current_entropy - target_entropy) > tolerance
#
#             return keep_going, current_sigma, upper_bound, lower_bound
#
#         keep_going, current_sigma, upper_bound, lower_bound = tf.while_loop(
#                     cond, body, (keep_going, current_sigma, upper_bound, lower_bound))
#         return current_sigma, i + 1
#
#     sigmas, _ = tf.map_fn(fn=lambda x: estimate_sigma(x[0], x[1]), elems=(negative_squared_distances, i))
#     return sigmas




def estimate_sigmas(distances,
                    tolerance=1e-5,
                    max_iter=50000,
                    lower_bound=1e-37,
                    upper_bound=np.float32(np.float32(2147483647.0) - np.float(100.0)),
                    target_perplexity=30.0):
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
    target_entropy = tf.math.log(target_perplexity)
    lower_bound_arr = tf.fill([n, 1], lower_bound)
    upper_bound_arr = tf.fill([n, 1], upper_bound)
    current_sigmas = tf.reshape(((lower_bound_arr + upper_bound_arr) / 2.), (-1, 1))
    done_mask = tf.fill([n], False)

    def cond(done_mask, *_):
        # print(tf.math.count_nonzero(done_mask))
        test = tf.math.reduce_all(done_mask)
        return not test

    def body(done_mask_inner, lower_bound_arr_inner, upper_bound_arr_inner, current_sigmas_inner):
        current_sigmas_inner = tf.reshape(((lower_bound_arr_inner + upper_bound_arr_inner) / 2.),
                                          (-1, 1))
        current_sigmas_squared_times_two = tf.square(current_sigmas_inner)
        distances_over_sigmas_squared_times_two = distances / current_sigmas_squared_times_two


        current_conditional_prob_of_neighbors = zeroed_softmax_axis(
            distances_over_sigmas_squared_times_two, axis=1)
        # print("prob: " + str(current_conditional_prob_of_neighbors[2779]))


        # print("dist: " + str(distances_over_sigmas_squared_times_two[2779]))
        # print("sigmas: " + str(current_sigmas_inner[2779]))
        # print("upper: " + str(upper_bound_arr_inner[2779]))
        # print("lower: " + str(lower_bound_arr_inner[2779]))

        current_entropy = compute_entropy(current_conditional_prob_of_neighbors)
        # tf.debugging.check_
        #
        # numerics(current_entropy, "hi")

        # print(tf.math.reduce_all(current_conditional_prob_of_neighbors < 0))

        # print("prob: " + str(tf.reduce_min(current_conditional_prob_of_neighbors[3748])))
        # print("perplexity: " + str(current_entropy[2779]))


        # print(tf.where(current_perplexity - target_perplexity < 5))
        # print(tf.reduce_max(current_perplexity, 0))
        # print(tf.reduce_min(current_perplexity, 0))
        # print(tf.math.argmax(current_entropy, 0))

        upper_mask = (current_entropy > target_entropy)
        upper_indices = tf.reshape(tf.where(upper_mask), (-1, 1))
        lower_mask = current_entropy < target_entropy
        lower_indices = tf.reshape(tf.where(lower_mask), (-1, 1))

        done_mask_inner = tf.abs(current_entropy - target_entropy) <= tolerance
        upper_bound_arr_inner = tf.tensor_scatter_nd_update(upper_bound_arr_inner, upper_indices,
                                                            current_sigmas_inner[upper_mask])
        lower_bound_arr_inner = tf.tensor_scatter_nd_update(lower_bound_arr_inner, lower_indices,
                                                            current_sigmas_inner[lower_mask])

        return done_mask_inner, lower_bound_arr_inner, upper_bound_arr_inner, current_sigmas_inner
    done_mask, lower_bound_arr, upper_bound_arr, current_sigmas_out = tf.while_loop(
        cond, body, (done_mask, lower_bound_arr, upper_bound_arr, current_sigmas))

    return current_sigmas_out


def compute_entropy(distribution):
    """
    Computes the perplexity of a distribution.
    :param distribution: The distribution to compute the perplexity of.
    :return: The perplexity.
    """
    next = 1e-8
    next = next + distribution
    blah = tf.math.log(next)


    aaah = tf.multiply(next, blah)


    entropy = -tf.reduce_sum(aaah, axis=1)
    # print(tf.where(tf.math.is_nan(entropy)))
    # print(entropy.shape)

    # print(entropy[3748])
    # print(entropy[484])
    # print(perplexity[484])
    # print(tf.reduce_max(next[484]))
    # print(tf.reduce_min(next[484]))
    # print(tf.reduce_max(blah[484]))
    # print(tf.reduce_min(blah[484]))
    # print(next[484, 0])
    # print(blah[484, 0])
    return entropy
