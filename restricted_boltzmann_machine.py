import tensorflow as tf


class RBM:
    """
    Restricted Boltzmann Machine -- especially designed for tsne. Do not use for other purposes
    without modification.
    """

    def __init__(self,
                 visible_units: int,
                 hidden_units: int,
                 linear: bool = False,
                 num_instances: int = 300,
                 n_iter: int = 30,
                 initial_momentum: float = 0.5,
                 final_momentum: float = 0.9,
                 weight_cost: float = 0.0002) -> None:
        """
        Create all of the weights and set up RBM to be linear or not. Defaults set up to follow
        parametric tsne matlab implementation.
        :param visible_units: The number of visible units in the RBM.
        :param hidden_units: The number of hidden units in the RBM.
        :param linear: Whether or not the RBM is linear.
        :param num_instances: The number of instances, in future should be batch size, but batches
        not currently supported.
        :param n_iter: Number of iterations to train for.
        :param initial_momentum: The initial momentum for contrastive divergence.
        :param final_momentum: The final momentum for contrastive divergence.
        :param weight_cost: The regularizer for weights.
        """
        super().__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.batch_size = num_instances
        self.n_iter = n_iter
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.weight_cost = weight_cost

        randn = tf.random_normal_initializer(mean=0.0, stddev=.1)

        self.activation = tf.nn.sigmoid
        self.sample = RBM.sample_bern
        self.learning_rate = .1
        if linear is True:
            self.learning_rate = .001
            self.activation = tf.keras.activations.linear
            self.sample = RBM.sample_gauss

        self.w = tf.Variable(
            randn(shape=[self.visible_units, self.hidden_units], dtype=tf.float32))
        self.visible_bias = tf.Variable(tf.zeros([1, self.visible_units]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([1, self.hidden_units]), dtype=tf.float32)

        self.delta_w = tf.Variable(
            tf.zeros([self.visible_units, self.hidden_units]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([1, self.visible_units]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([1, self.hidden_units]), dtype=tf.float32)

    def fit(self, x):
        """
        Fits the RBM using Contrastive Divergence.
        :param x: The features to train on.
        :return: The trained model. (Allows chaining)
        """
        for i in range(self.n_iter):
            if i <= 5:
                momentum = self.initial_momentum
            else:
                momentum = self.final_momentum

            hidden_prob_one = self.activation(tf.add(tf.matmul(x, self.w), self.hidden_bias))
            hidden_states = self.sample(hidden_prob_one)
            vis_prob = tf.nn.sigmoid(
                tf.add(tf.matmul(hidden_states, self.w, transpose_b=True), self.visible_bias))
            hidden_prob_two = self.activation(tf.add(tf.matmul(vis_prob, self.w), self.hidden_bias))

            pos_prods = tf.matmul(x, hidden_prob_one, transpose_a=True)
            neg_prods = tf.matmul(vis_prob, hidden_prob_two, transpose_a=True)

            self.delta_w = tf.add(
                tf.multiply(momentum, self.delta_w),
                tf.multiply(self.learning_rate,
                            tf.subtract(
                                tf.divide(tf.subtract(pos_prods, neg_prods), self.batch_size),
                                tf.multiply(self.weight_cost, self.w))))

            self.delta_hidden_bias = tf.add(
                tf.multiply(momentum, self.delta_hidden_bias),
                tf.multiply(tf.divide(self.learning_rate, self.batch_size),
                            tf.subtract(
                                tf.reduce_sum(hidden_prob_one, 0),
                                tf.reduce_sum(hidden_prob_two, 0))))

            self.delta_visible_bias = tf.add(
                tf.multiply(momentum, self.delta_visible_bias),
                tf.multiply(tf.divide(self.learning_rate, self.batch_size),
                            tf.subtract(
                                tf.reduce_sum(x, 0),
                                tf.reduce_sum(vis_prob, 0))))

            self.w.assign_add(self.delta_w)
            self.hidden_bias.assign_add(self.delta_hidden_bias)
            self.visible_bias.assign_add(self.delta_visible_bias)
        return self

    def transform(self, x):
        """
        DANGER. Since this RBM is only meant to be stacked this transforms features into what the
        next layer expects.
        :param x: The features to transform.
        :return: The transformed features. (No chaining)
        """
        return self.activation(tf.add(tf.matmul(x, self.w), self.hidden_bias))

    def get_w(self):
        """
        Gets the weight matrix from this RBM.
        :return: The weight matrix.
        """
        return self.w

    def get_hidden_bias(self):
        """
        Gets the bias used to transform the visible to the hidden.
        :return: The hidden bias.
        """
        return self.hidden_bias

    @staticmethod
    def sample_bern(hidden_prob):
        """
        Samples from hidden probabilities based on uniform.
        :param hidden_prob: The hidden probability.
        :return: The samples.
        """
        rand = tf.random.uniform(tf.shape(hidden_prob), minval=0, maxval=1)
        return tf.cast((hidden_prob > rand), tf.float32)

    @staticmethod
    def sample_gauss(hidden_prob):
        """
        Samples from hidden probabilities based on gaussian.
        :param hidden_prob: The hidden probability.
        :return: The samples.
        """
        rand = tf.random.normal(tf.shape(hidden_prob), mean=0, stddev=1)
        return tf.add(hidden_prob, rand)
