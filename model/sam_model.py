import re
import os

import numpy as np
import tensorflow as tf

from model.encoder import Encoder
from model.decoder import Decoder
from utils.image_embeddings import ResNet
from utils.rnn_model import rnn_placeholders


class SamModel():
    def __init__(self, params, data=None, comm_max_len=None, mode="train_eval"):
        assert mode in ("train_eval", "out_gen"), "train_eval or out_gen"
         # Image
        self._img_size = [224, 224]  # Resnet input size
        # Generation
        self._temperature = 0.6
        self._gen_method = "greedy"
        # Model parameters
        self._com_hid_dim = 512
        self._seqattr_hid_dim = 512
        self._embed_dim = 1024
        if mode == "train_eval":
            self._data = data["train"]
            # Val/test
            self._val_data = data["val"]
            # Model training parameters
            self._n_comms = params.num_comms  # Number of comments for one Post
            self._batch_size = params.batch_size
            self._gpu = params.gpu
            # Optimization
            self._n_epochs = params.epochs
            self._learning_rate = params.lr
            self._dec_dropout = 0.9
            # Checkpoints
            self._resnet_ckpt = "./utils/resnet_v1_imagenet/model.ckpt-257706"
            self._ckpt_dir = params.checkpoint_dir
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
        elif mode == "out_gen":  # For usage as ouside API
            tf.reset_default_graph()
            self._gpu = params.gpu
            self._data = data
            self._resnet_ckpt = "./utils/resnet_v1_imagenet/model.ckpt-257706"
            self._ckpt_dir = params.checkpoint_dir
            ckpt_name = params.checkpoint
            self._model_ckpt = os.path.join(self._ckpt_dir, ckpt_name)
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
        # Attributes, categorical
        self._urls = tf.placeholder(tf.int32, [None], name="urls")
        # Attributes, sequential
        self._postname = tf.placeholder(
            tf.int32, [None, None], name="postname")
        # Image paths
        self._impaths = tf.placeholder(tf.string, [None], "impaths")

    def __set_inference_ph(self):
        self._comm_inputs = tf.placeholder(
            tf.int32, [None], name="comments_x")
        self._urls = tf.placeholder(tf.int32, [None], name="urls")
        self._postname = tf.placeholder(
            tf.int32, [None, None], name="postname")
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
        # im_features = resnet(images, is_training)
        im_features = resnet(images, is_training, ret_pre_pool=True)
        # [batch_size, 7 x 7, 2048]
        im_features = tf.reshape(
            im_features, [self._batch_size, 7 * 7, resnet.final_size]) 
        if self._n_comms > 1:
            # Trick to avoid expensive forward passes through ResNet
            # [batch_size*5, im_feature_dim]
            # im_features = tf.tile(
            #     tf.expand_dims(im_features, 1), [1, self._n_comms, 1])
            im_features = tf.tile(
                tf.expand_dims(im_features, 1), [1, self._n_comms, 1, 1])
            # im_features = tf.reshape(
            #     im_features, [-1, self._img_embed_size])
            im_features = tf.reshape(
                im_features, 
                [self._batch_size * self._n_comms,
                 7 * 7,
                 self._img_embed_size])
        self._im_features = im_features

    def __initial_gru(self, features):
        with tf.variable_scope('initial_gru'):
            features = tf.reduce_mean(features, 1)  # Used if using conv
            h = tf.nn.tanh(tf.layers.dense(features, self._com_hid_dim))
            return h

    def __build_layers(self):
        # For every post word calculate attention
        post_vocab = self._data.dictionaries["PostName"].vocab_size
        comm_vocab = self._data.dictionaries["Comments"].vocab_size
        seq_attr_ecoder = Encoder(
            "postname_enc", self._seqattr_hid_dim, self._embed_dim, post_vocab)
        comm_decoder = Decoder(
            "comments_dec", self._com_hid_dim, 
            self._embed_dim, comm_vocab, self._dec_dropout)
        dec_state = None
        # TODO: change static RNN into dynamic rnn, use AttentionCellWrapper
        loss = 0.0
        pad_word = self._data.dictionaries["Comments"].word2idx["<PAD>"]
        mask = tf.to_float(
            tf.not_equal(self._comm_labels, pad_word))
        batch_size = tf.shape(self._comm_inputs)[0]
        # img_emb = tf.layers.dense(
        #     self._im_features, self._embed_dim, name="feature_proj")
        # img_emb = tf.Print(img_emb, [tf.shape(img_emb)], first_n=1)
        for t in range(self._comm_max_len):
            # if dec_state == None:
            #     # For an original dec state for attention
            #     dec_state = comm_decoder.rnn_cell.zero_state(
            #     batch_size=batch_size, dtype=tf.float32)
            if dec_state == None:
                dec_state = self.__initial_gru(self._im_features)
            enc_states = seq_attr_ecoder(self._postname)
            enc_state_att = self.__attention_layer(enc_states, dec_state, "1")
            # img_embed_att = self.__attention_layer(
            #     tf.expand_dims(self._im_features, 1), 
            #     dec_state, "3")
            img_embed_att =  self.__attention_layer(
                self._im_features, dec_state, "3")
            # [batch_size, embed_dim]
            # urls = self.__embedding_layer(
            #     "urls_embed", self._urls, self._embed_dim, 
            #     self._data.dictionaries["Urls"].vocab_size)
            # Concatenate
            enc_state_att = tf.layers.dense(
                enc_state_att, self._embed_dim, 
                name="enc_emb_dense", reuse=t!=0)
            img_embed_att = tf.layers.dense(
                img_embed_att, self._embed_dim,
                name="img_emb_dense", reuse=t!=0
            )
            # attr_list = [urls, enc_state_att, img_embed_att]
            attr_list = [enc_state_att, img_embed_att]
            attributes = tf.stack(attr_list, 1)
            attr_att = self.__attention_layer(attributes, dec_state, "2")
            # Decoder, input [batch_size, embed_dim + attrib_embed_dim]
            logits, dec_state = comm_decoder(
                self._comm_inputs[:, t], attr_att, dec_state, t)
            loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self._comm_labels[:, t], logits=logits)*mask[:, t])
        self.loss = loss / tf.to_float(batch_size)
    
    def __build_inference_graph(self):
        # For every post word calculate attention
        post_vocab = self._data.dictionaries["PostName"].vocab_size
        comm_vocab = self._data.dictionaries["Comments"].vocab_size
        seq_attr_ecoder = Encoder(
            "postname_enc", self._seqattr_hid_dim, self._embed_dim, post_vocab)
        self._comm_decoder = Decoder(
            "comments_dec", self._com_hid_dim, self._embed_dim, comm_vocab, 1.0)
        # self._img_emb = tf.layers.dense(
        #     self._im_features, self._embed_dim, name="feature_proj")
        # For an original dec state for attention
        dec_state = self.__initial_gru(self._im_features)
        self._state = rnn_placeholders(dec_state)
        enc_states = seq_attr_ecoder(self._postname)
        enc_state_att = self.__attention_layer(enc_states, self._state, "1")
        img_embed_att =  self.__attention_layer(
            self._im_features, dec_state, "3")
        # [batch_size, embed_dim]
        # urls = self.__embedding_layer(
        #     "urls_embed", self._urls, self._embed_dim, 
        #     self._data.dictionaries["Urls"].vocab_size)
        enc_state_att = tf.layers.dense(
            enc_state_att, self._embed_dim, name="enc_emb_dense")
        img_embed_att = tf.layers.dense(
                img_embed_att, self._embed_dim,
                name="img_emb_dense")
        # attr_list = [urls, enc_state_att]
        attr_list = [enc_state_att, img_embed_att]
        attributes = tf.stack(attr_list, 1)
        attr_att = self.__attention_layer(attributes, self._state, "2")
        attr_att += img_embed_att
        logits, self._dec_state = self.__forward(
            self._comm_inputs, attr_att, self._state, 0)
        self._logits = tf.nn.softmax(logits)

    def __forward(self, inputs, attrib_vectors, state, t):
        return self._comm_decoder(inputs, attrib_vectors, state, t)

    def __embedding_layer(self, name, inputs, embed_dim, vocab_size):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            embeddings = tf.get_variable(
                "embed", [vocab_size, embed_dim], tf.float32)
            return tf.nn.embedding_lookup(embeddings, inputs)

    # def __attention_weight(self, value_one, value_two, name):
    #     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    #         # Multiplicative attention
    #         att_w = tf.get_variable(
    #             "M_{}".format(name), 
    #             [value_one.shape[1], self._batch_size], tf.float32)
    #         att = tf.matmul(value_one, att_w)
    #         att = tf.matmul(att, value_two)
    #         # [batch_size, hidden_size]
    #         return tf.nn.softmax(att)

    def __attention_layer(self, attention_states, decoder_state, name):
        with tf.variable_scope("att_layer{}".format(name), reuse=tf.AUTO_REUSE):
            # hidden: [batch_size, seq_len, hidden_dim]
            # attn_length = attention_states.get_shape()[1].value
            attn_length = tf.shape(attention_states)[1]
            attn_size = attention_states.get_shape()[2].value
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            hidden = tf.reshape(
                attention_states, [-1, attn_length, 1, attn_size])
            attention_vec_size = attn_size  # Size of query vectors for attention.
            k = tf.get_variable(
                "M_{}".format(name), [1, 1, attn_size, attention_vec_size])
            v = tf.get_variable("att_w_{}".format(name), [attention_vec_size])
            att_feat = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            # Additive attention w * (W1*h + W2*s)
            y = tf.layers.dense(decoder_state, attention_vec_size)
            s = tf.reduce_sum(v * tf.nn.tanh(att_feat + y), [2, 3])
            att = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(
                tf.reshape(att, [-1, attn_length, 1, 1]) * hidden, [1, 2])
        return tf.reshape(d, [-1, attn_size])

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
    
    def __train_on_batch(self, impaths, urls, comms_l, pname, other_attrs, mode):
        feed = {
                self._urls: urls,
                self._comm_inputs: comms_l[0],
                self._comm_labels: comms_l[1],
                self._postname: pname}
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
                for impaths, urls, comms_l, pname, other_attrs in b_gen(
                    self._batch_size):
                    cur_loss, summary = self.__train_on_batch(
                        impaths, urls, comms_l, pname, other_attrs, "train")
                    summary_writer.add_summary(summary, step_global)
                    step_global += 1
                for impaths, urls, comms_l, pname, other_attrs in val_bgen(
                    self._batch_size):
                    val_loss, _= self.__train_on_batch(
                        impaths, urls, comms_l, pname, other_attrs, "eval")
                print("Epoch: {}".format(e))
                print("Train Loss: {}".format(cur_loss))
                print("Validation loss: {}".format(val_loss))
                if not os.path.exists(self._ckpt_dir):
                    os.makedirs(self._ckpt_dir)
                model_saver.save(sess, self._model_ckpt)
                print("Model weights saved in : ", self._model_ckpt)
                print("=============================================")
            print("Training finished")
    
    def __predict_on_batch(self, images, urls, pnames, other_attrs, ret_json=True):
        # TODO: build prediction graph, save into json file
        comm_dict = self._data.dictionaries["Comments"]
        if len(urls.shape) == 0:  # To avoid problems if only one url in batch
            urls = np.tile(urls, 1)
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
                    self._urls: np.tile(np.array(urls[i]), 1),
                    self._comm_inputs: new_word,
                    self._postname: np.expand_dims(pnames[i], 0)
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
                    gen_comms = self.__predict_on_batch(
                        impaths, urls, pname, other_attrs)
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
        # sess.run(tf.global_variables_initializer())
        # restore
        resnet_saver.restore(sess, self._resnet_ckpt)
        model_saver.restore(sess, checkpoint)
        self.sess = sess

    def predict_on_batch(self, image_path, url, pname, other_attrs=None):
        # Tokenize
        self._comm_max_len = 30
        url = np.array(self._data.dictionaries["Urls"].transform(url))
        pname = np.array([self._data.dictionaries["PostName"].transform(pname)])
        url = np.tile(url, 1)
        image_path = np.tile(np.array(image_path), 1)
        # Will be used for external API
        return self.__predict_on_batch(
            image_path, url, pname, other_attrs, False)

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
