import os
import re

import numpy as np
import tensorflow as tf

from model.encoder import Encoder
from model.decoder import Decoder
from utils.image_embeddings import ResNet
from utils.rnn_model import rnn_placeholders


class NicModel():
    def __init__(self, params, data, comm_max_len=None, mode="train_eval"):
        assert mode in ("train_eval", "out_gen"), "train_eval or out_gen"
        # Generation
        self._temperature = 0.6
        self._gen_method = "greedy"
        # Image
        self._img_size = [224, 224]  # Resnet input size
        self._resnet_ckpt = "./utils/resnet_v1_imagenet/model.ckpt-257706"
        self._gpu = params.gpu
        self._ckpt_dir = params.checkpoint_dir
        # Model parameters
        self._com_hid_dim = 1024
        self._seqattr_hid_dim = 512
        self._embed_dim = 512
        if mode == "train_eval":
            self._data = data["train"]
            # Val/test
            self._val_data = data["val"]
            self._n_comms = params.num_comms  # Number of comments for one Post
            self._batch_size = 32
            self._dec_dropout = 0.9
            # Optimization
            self._n_epochs = params.epochs
            self._learning_rate = params.lr
            # Checkpoints
            ckpt_name = params.checkpoint
            model_ckpt = "{}_{}_{}".format(
                ckpt_name, self._n_epochs, self._learning_rate)
            self._model_ckpt = os.path.join(self._ckpt_dir, model_ckpt)
            # Set max len for able to use static
            self._comm_max_len = comm_max_len
            # Summaries
            self._logdir = params.logdir
            # Define model
            self.__set_placeholders()
            self.__img_embeddings()
            self.__build_layers()
        elif mode == "out_gen":
            tf.reset_default_graph()
            self._data = data
            ckpt_name = params.checkpoint
            self._model_ckpt = os.path.join(self._ckpt_dir, ckpt_name)
            # Technical settings
            self._n_comms = 1  # To avoid tiling of images
            self._batch_size = 1
            # Inference graph
            self.__set_inference_ph()
            self.__img_embeddings()
            self.__build_inference_graph()
            self.__init_gen_session()

    def __set_placeholders(self):
        # Base placeholders: comments, urls, labels
        self._comm_inputs = tf.placeholder(
            tf.int32, [None, self._comm_max_len], name="comments_x")
        self._comm_labels = tf.placeholder(
            tf.int32, [None, self._comm_max_len], name="comments_y")
        # Image paths
        self._impaths = tf.placeholder(tf.string, [None], "impaths")

    def __set_inference_ph(self):
        self._comm_inputs = tf.placeholder(
            tf.int32, [None], name="comments_x")
        self._impaths = tf.placeholder(tf.string, [None], "impaths")

    def __img_embeddings(self):
        import multiprocessing
        def _parse_function(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image_resized = tf.image.resize_images(
                image_decoded, self._img_size)
            return image_resized
        dataset = tf.data.Dataset.from_tensor_slices(self._impaths)
        dataset = dataset.map(
            _parse_function, num_parallel_calls=multiprocessing.cpu_count())
        # TODO: variable batch sizes to match tensor shape?
        dataset = dataset.batch(self._batch_size)
        iterator = dataset.make_initializable_iterator()
        self._iterator = iterator
        images = iterator.get_next()
        resnet = ResNet(50)  # Have checkpoints fo ResNet 50
        self._img_embed_size = resnet.final_size
        is_training = tf.constant(False)
        # TODO: subjected to change, maybe use last conv layer as features?
        im_features = resnet(images, is_training)
        if self._n_comms > 1:
            # Trick to avoid expensive forward passes through ResNet
            # [batch_size*5, im_feature_dim]
            im_features = tf.tile(
                tf.expand_dims(im_features, 1), [1, self._n_comms, 1])
            im_features = tf.reshape(
                im_features, [-1, self._img_embed_size])
        self._im_features = im_features

    def __build_layers(self):
        # For every post word calculate attention
        comm_vocab = self._data.dictionaries["Comments"].vocab_size
        comm_decoder = Decoder(
            "comments_dec", self._com_hid_dim, 
            self._embed_dim, comm_vocab, self._dec_dropout)
        pad_word = self._data.dictionaries["Comments"].word2idx["<PAD>"]
        mask = tf.to_float(
            tf.not_equal(self._comm_labels, pad_word))
        img_emb = tf.layers.dense(
            self._im_features, self._embed_dim, name="feature_proj")
        dec_state = None
        _, dec_state = comm_decoder(
            img_emb, None, dec_state, -1)
        logits, dec_state = comm_decoder.dynamic_call(
            self._comm_inputs,
            dec_state)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._comm_labels, logits=logits)*mask)
    
    def __build_inference_graph(self):
        # For every post word calculate attention
        comm_vocab = self._data.dictionaries["Comments"].vocab_size
        self._comm_decoder = Decoder(
            "comments_dec", self._com_hid_dim, self._embed_dim, comm_vocab, 1.0)
        self._dec_state = None
        self._img_emb = tf.layers.dense(
            self._im_features, self._embed_dim, name="feature_proj")
        _, state = self.__forward(
            self._img_emb, None, None, -1)
        self._state = rnn_placeholders(state)
        logits, self._dec_state = self.__forward(
            self._comm_inputs, None, self._state, 0)
        self._logits = tf.nn.softmax(logits)

    def __forward(self, inputs, attrib_vectors, state, t):
        return self._comm_decoder(inputs, attrib_vectors, state, t)

    def __embedding_layer(self, name, inputs, embed_dim, vocab_size):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            embeddings = tf.get_variable(
                "embed", [vocab_size, embed_dim], tf.float32)
            return tf.nn.embedding_lookup(embeddings, inputs)

    def __optimize(self):
        self._resnet_var = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_model")
        tr_variables = tf.trainable_variables()
        trainable_variables = [
            var for var in tr_variables if var not in self._resnet_var]
        self._trainable_variables = trainable_variables
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        print(trainable_variables)
        return optimizer.minimize(self.loss, var_list=trainable_variables)
    
    def __set_summaries(self):
        # tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", self._learning_rate)
        merged = tf.summary.merge_all()
        self._summaries = merged

    def __sess_config(self):
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(
            visible_device_list=self._gpu))
        return config
    
    def __train_on_batch(self, impaths, comms_l, mode):
        feed = {
                self._comm_inputs: comms_l[0],
                self._comm_labels: comms_l[1]
                }
        # Initialize iterator for usage with placeholders
        self.sess.run(self._iterator.initializer, {self._impaths: impaths})
        if mode == "train":
            loss, _ = self.sess.run([self.loss, self._optimize], feed)
        elif mode == "eval":
            loss = self.sess.run(self.loss, feed)
        # tf.summary.scalar("loss", loss)
        summary = self.sess.run(self._summaries, feed)
        return loss, summary

    def train(self):
        self._optimize = self.__optimize()
        self.__set_summaries()
        b_gen = self._data.batch_generator
        val_bgen = self._val_data.batch_generator
        model_saver = tf.train.Saver(self._trainable_variables)
        resnet_saver = tf.train.Saver(self._resnet_var)
        with tf.Session(config=self.__sess_config()) as sess:
            self.sess = sess
            summary_writer = tf.summary.FileWriter(self._logdir, sess.graph)
            sess.run(tf.global_variables_initializer())
            # Restore Resnet Weights
            resnet_saver.restore(sess, self._resnet_ckpt)
            step_global = 0
            for e in range(self._n_epochs):
                for impaths, _, comms_l, _, _ in b_gen(
                    self._batch_size):
                    cur_loss, summary = self.__train_on_batch(
                        impaths, comms_l, "train")
                    summary_writer.add_summary(summary, step_global)
                    step_global += 1
                for impaths, urls, comms_l, pname, other_attrs in val_bgen(
                    self._batch_size):
                    val_loss, _= self.__train_on_batch(
                        impaths, comms_l, "eval")
                print("Epoch: {}".format(e))
                print("Train Loss: {}".format(cur_loss))
                print("Validation loss: {}".format(val_loss))
                if not os.path.exists(self._ckpt_dir):
                    os.makedirs(self._ckpt_dir)
                model_saver.save(sess, self._model_ckpt)
                print("Model weights saved in : ", self._model_ckpt)
                print("=============================================")
            print("Training finished")
    
    def __init_gen_session(self):
        # Restore from checkpoints
        self._resnet_var = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_model")
        tr_variables = tf.trainable_variables()
        trainable_variables = [
            var for var in tr_variables if var not in self._resnet_var]
        model_saver = tf.train.Saver(trainable_variables)
        resnet_saver = tf.train.Saver(self._resnet_var)
        res_folder = "./results"
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        checkpoint = self._model_ckpt
        sess = tf.InteractiveSession(config=self.__sess_config())
        # restore
        resnet_saver.restore(sess, self._resnet_ckpt)
        model_saver.restore(sess, checkpoint)
        self.sess = sess

    def __predict_on_batch(self, images, ret_json=True):
        # TODO: build prediction graph, save into json file
        comm_dict = self._data.dictionaries["Comments"]
        sequences = []
        for i, image in enumerate(images):
            sequence = []
            # Pad sequence
            cur_state = None
            for t in range(self._comm_max_len):
                if t == 0:
                    new_word = np.tile(
                        np.array([comm_dict.word2idx["<BOS>"]]), 1)
                else:
                    new_word = np.tile(new_word, 1)
                feed_dict = {
                    self._comm_inputs: new_word
                }
                if cur_state is not None:
                    feed_dict.update({self._state: cur_state})
                self.sess.run(
                    self._iterator.initializer,
                    {self._impaths: np.tile(image, 1)})
                logits, cur_state = self.sess.run(
                    [self._logits, self._dec_state], feed_dict)
                next_word_probs = logits.ravel()
                t = self._temperature
                next_word_probs = next_word_probs**(
                    1/t) / np.sum(next_word_probs**(1/t))
                new_word = np.argmax(next_word_probs)
                # new_word = np.argmax(logits)
                sequence.append(new_word)
                if new_word == comm_dict.word2idx["<EOS>"]:
                    break
            sequence = " ".join(
                [comm_dict.idx2word[idx] for idx in sequence[:-1]])
            if ret_json:
                imid = "".join(re.findall("[0-9]", image))
                cap_dict = {
                    'image_id': int(imid),
                    'caption': sequence}
                sequences.append(cap_dict)
            else:
                sequences.append(sequence)
            print(sequences[i])
        return sequences                       
    
    def eval_test(self, checkpoint, res_fn, data=None, 
    data_dict=None, save_into_file=True, save_fn=None, checkp_dir=None):
        if data is None and data_dict is None:
            raise("Must provide data class object or data dict")
        # TODO: first use data class, then finish for exteranal picture data
        tf.reset_default_graph()
        # Initiate model
        self.__set_inference_ph()
        self._n_comms = 1  # To avoid tiling of images
        self._batch_size = 1
        self.__img_embeddings()
        self.__build_inference_graph()
        # Restore from checkpoints
        self._resnet_var = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_model")
        tr_variables = tf.trainable_variables()
        trainable_variables = [
            var for var in tr_variables if var not in self._resnet_var]
        model_saver = tf.train.Saver(trainable_variables)
        resnet_saver = tf.train.Saver(self._resnet_var)
        res_folder = "./results"
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        if checkp_dir:
            checkpoint = os.path.join(checkp_dir, checkpoint)
        with tf.Session(config=self.__sess_config()) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            # restore
            resnet_saver.restore(sess, self._resnet_ckpt)
            model_saver.restore(sess, checkpoint)
            if data is not None:
                b_gen = data.batch_generator
                # Save to ./res_folder/res_fn.json
                comms_list = []
                for impaths, urls, _, pname, other_attrs in b_gen(1):
                    gen_comms = self.__predict_on_batch(impaths)
                    comms_list.extend(gen_comms)
                
            if save_into_file:
                import json
                gen_file = os.path.join(res_folder, res_fn)
                if os.path.exists(gen_file):
                    print("")
                    os.remove(gen_file)
                with open (gen_file, "w") as wf:
                    json.dump(comms_list, wf)
                print("Results saved into: ", gen_file)

    def predict_on_batch(self, image_path):
        # Tokenize
        self._comm_max_len = 30
        image_path = np.tile(np.array(image_path), 1)
        # Will be used for external API
        return self.__predict_on_batch(image_path, False)

    def test_datagen(self):
        b_gen = self._data.batch_generator
        for impaths, urls, comms_l, pname, other_attrs in b_gen(
                    self._batch_size):
            print(impaths.shape)
            print(comms_l[0].shape)
            print(urls.shape)
            print(pname.shape)
            comm, comm_l = comms_l[0], comms_l[1]
            comm_dict = self._data.dictionaries["Comments"]
            for com, coml in zip(comm, comm_l):
                print(" ".join([comm_dict.idx2word[wd] for wd in com if not (wd==0)]))
            exit()