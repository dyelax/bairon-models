import tensorflow as tf
import numpy as np
import os

from utils.data_processing import preprocess, postprocess, get_random_word
from utils.tfultils import sample

class WordModel:
    def __init__(self, args, vocab):
        self.args = args
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        self.sess = tf.Session()

        self._build_graph()

        print 'Init variables...'
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

        # if load path specified, load a saved model
        if args.load_path is not None:
            self.saver.restore(self.sess, self.args.load_path)
            print 'Model restored from ' + self.args.load_path

    def _build_graph(self):
        with tf.name_scope('Word_Model'):
            with tf.name_scope('Inputs'):
                # inputs and targets are 2D tensors of shape (batch_size, seq_len)
                self.inputs = tf.placeholder(tf.int32, [None, None])
                self.targets = tf.placeholder(tf.int32, [None, None])
                self.batch_size = tf.shape(self.inputs)[0]

                self.keep_prob = tf.placeholder(tf.float32)

                logits, state = self._compute(self.inputs)

            with tf.name_scope('Optimization'):
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.reshape(self.targets, [-1]))) / self.args.seq_len

                self.global_step = tf.Variable(0, trainable=False, name='global_step')

                self.lr = tf.train.exponential_decay(
                    self.args.lr, self.global_step, self.args.lr_decay_steps, self.args.lr_decay_rate)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr,
                                                        name='optimizer')
                self.train_op = self.optimizer.minimize(self.loss,
                                                        global_step=self.global_step,
                                                        name='train_op')

            with tf.name_scope('Inference'):
                # Create the generated sequence tensor as a batch of size 1, and the primer input
                # sequence length + the generated sequence length
                self.gen_seq = self.inputs

                # We already generated the first new word in the process of setting the state from
                # the primer input
                if self.args.argmax:
                    pred_word = tf.cast(tf.argmax(logits[-1]), tf.int32)
                else:
                    pred_word = sample(logits)[-1][0]
                self.gen_seq = tf.concat([self.gen_seq, tf.reshape(pred_word, [1, 1])], axis=1)

                def gen_next_word(i, gen_seq, state):
                    """
                    Generates the next word and adds it to the sequence. Used as the body of the
                    generation loop.

                    :param i: The index of the word we are generating in gen_seq
                    :param gen_seq:
                    :param state:
                    :return:
                    """
                    logits, state = self._compute(gen_seq, initial_state=state)
                    if self.args.argmax:
                        pred_word = tf.cast(tf.argmax(logits[-1]), tf.int32)
                    else:
                        pred_word = sample(logits)[-1][0]
                    gen_seq = tf.concat([gen_seq, tf.reshape(pred_word, [1, 1])], axis=1)

                    return i + 1, gen_seq, state

                # The run condition of the loop
                cond = lambda i, gen_seq, state: i < self.args.max_gen_len

                state_shapes = tuple([tf.TensorShape([None, self.args.cell_size]) for _ in state])
                _, self.gen_seq, state = tf.while_loop(cond, gen_next_word,
                                                       loop_vars=(1, self.gen_seq, state),
                                                       shape_invariants=(
                                                           tf.TensorShape([]),
                                                           tf.TensorShape([None, None]),
                                                           state_shapes)
                                                       )


    def _compute(self, inputs, initial_state=None):
        """
        Compute next-word predictions using the GRU network.

        :param inputs: The inputs for which to predict the next words.
        :param initial_state: The initial state of the GRU.

        :return: A tuple, (logits, softmax_probs, state)
        """
        with tf.variable_scope('Variables', reuse=tf.AUTO_REUSE):
            # Fully connected layer from the output of the GRU
            initializer = tf.contrib.layers.xavier_initializer()
            self.ws = tf.get_variable(
                'ws', (self.args.cell_size, self.vocab_size), initializer=initializer)
            self.bs = tf.get_variable(
                'bs', (self.vocab_size,), initializer=initializer)

            with tf.device('/cpu:0'):  # put on CPU to parallelize for faster training/
                self.embeddings = tf.get_variable(
                    'embeddings', [self.vocab_size, self.args.cell_size], initializer=initializer)

                # get embeddings for all input words
                input_embeddings = tf.nn.embedding_lookup(self.embeddings, inputs)

            self.cell = tf.nn.rnn_cell.GRUCell(self.args.cell_size)
            self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell,
                                                      input_keep_prob=self.keep_prob,
                                                      output_keep_prob=self.keep_prob,
                                                      state_keep_prob=self.keep_prob, )
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self.args.num_layers)

            if initial_state is None:
                initial_state = self.cell.zero_state(self.batch_size, tf.float32)
            gru_outputs, state = tf.nn.dynamic_rnn(cell=self.cell,
                                                            inputs=input_embeddings,
                                                            dtype=tf.float32,
                                                            initial_state=initial_state)

            gru_outputs_flat = tf.reshape(tf.concat(gru_outputs, axis=1),
                                          [-1, self.args.cell_size])

            logits = tf.matmul(gru_outputs_flat, self.ws) + self.bs

        return logits, state

    def generate(self, primer=None, save_path=None):
        """
        Generate a sequence of words given priming text.

        :param primer: An initial string of text on which to condition the
                       generated sequence.

        :return: A sequence of generated text.
        """
        if primer is None:
            primer = get_random_word(self.vocab)

        primer_is = preprocess(primer, self.vocab)

        feed_dict = {self.inputs: np.array([primer_is]),
                     self.keep_prob: 1}
        gen_seq_is = self.sess.run(tf.squeeze(self.gen_seq), feed_dict=feed_dict)

        gen_text = postprocess(gen_seq_is, self.vocab)
        print gen_text

        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(gen_text)

    def train_step(self, inputs, targets):
        """
        Perform one training step on the model

        :param inputs: A batch of word sequences.
        :param targets: A batch of target word sequences (inputs shifted by one word).

        :return: The global step.
        """

        feed_dict = {self.inputs: inputs,
                     self.targets: targets,
                     self.keep_prob: self.args.keep_prob}
        global_step, loss, lr, _ = self.sess.run([self.global_step,
                                              self.loss,
                                              self.lr,
                                              self.train_op],
                                             feed_dict=feed_dict)

        print 'Step: %d | lr: %f | loss: %f' % (global_step, lr, loss)
        if (global_step - 1) % self.args.model_save_freq == 0:
            print 'Saving model...'
            self.saver.save(self.sess, os.path.join(self.args.save_dir, 'model'),
                            global_step=global_step)

        if (global_step - 1) % self.args.inference_freq == 0:
            self.generate(save_path=os.path.join(self.args.save_dir, str(global_step) + '.txt'))

        return global_step
