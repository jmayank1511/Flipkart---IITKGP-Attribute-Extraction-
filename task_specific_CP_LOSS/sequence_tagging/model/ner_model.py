import numpy as np
import os
import tensorflow as tf


from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.idx_to_word = {idx: word for word, idx in self.config.vocab_words.items()}
        self.task =0 


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)



    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """


        with tf.variable_scope("bi-lstm"):

            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
            #print "Main output"
            #print output

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])
            #print "Main W"
            #print W

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            #print "Main B"
            #print b

            nsteps = tf.shape(output)[1]
            #print "nsteps"
            #print nsteps
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            #print "New output"
            #print output
            pred = tf.matmul(output, W) + b
            #print "pred"
            #print pred
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])
            #print "logit"
            #print self.logits
            #print (self.logits.eval())
            #with tf.Session():
               # print (pred.eval())


    


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.sequence_lengths)
            #init_op = tf.initialize_all_variables()
            #with tf.Session() as sess:
                #sess.run(init_op) 
                #print (sess.run(self.labels))
            #print (self.labels)

            #self.cp_loss = coupling_loss(self.logits, self.labels, self.sequence_lengths)
            
            self.trans_params = trans_params 
            self.loss = tf.reduce_mean(-log_likelihood)

        tf.summary.scalar("loss", self.loss)




    def loadGloveModel(gloveFile):
        print ("Loading Glove Model")
        with open(gloveFile, encoding="utf8" ) as f:
            content = f.readlines()
            model = {}
            for line in content:
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
            print ("Done.",len(model)," words loaded!")
            return model



    def attention(dress, jean):
        As_left = tf.placeholder(tf.float32, shape=[None, 300], name="left")
        Bs_left = tf.placeholder(tf.float32, shape=[None, 300], name="left")
        w_omega_left = tf.Variable(tf.random_normal([1, 300], stddev=0.1),name="left")
        with tf.name_scope('left'):
            v_left = tf.matmul(As_left, Bs_left) 
            temp_left = tf.multiply(v_left, w_omega_left)
            output_left = tf.nn.softmax(temp_left, name='alphas_left')


        As_right = tf.placeholder(tf.float32, shape=[None, 300], name="right")
        Bs_right = tf.placeholder(tf.float32, shape=[None, 300], name="right")
        w_omega_right = tf.Variable(tf.random_normal([1, 300], stddev=0.1),name="right")
        with tf.name_scope('right'):
            v_right = tf.matmul(As_right, Bs_right) 
            temp_right = tf.multiply(v_right, w_omega_right)
            output_right = tf.nn.softmax(temp_right, name='alphas_right') 


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #A = np.random.randint(5, size=(3, 3))
            #B = np.random.randint(5, size=(3, 3))
            left = sess.run(output_left,feed_dict={As_left:dress,Bs_left:jean.T})
            right = sess.run(output_right,feed_dict={As_right:jean,Bs_right:dress.T})
            #print(left)
            #print(right)
        return left,right



  
    def add_coupling_loss(self):

        f = open("dress_list.txt",'r')
        g = open("tags.txt", 'r')
        tag_list =[]

        cp_loss =0

        for lines in g.readlines():
            ls = lines.split("\n")
            tag_list.append(lines)

        #print(self.labels)
        example =[]
        count =[]
        my_lambda =[]

        for lines in f.readlines():
            ls = lines.strip().split("\t")
            example.append(ls[0])
            count.append(ls[1])
            #my_lambda.append(ls[2])

        val = tf.map_fn(lambda x: (x, x), self.labels, dtype=(tf.int32, tf.int32))
        
        #with tf.Session() as sess:
            #print(sess.run(val),feed_dict=fd)


        #fp = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
        file = "data/glove.6B/glove.6B.300d"

        for i in range (0,len(tag_list)):
            if i in val:
                st = example[i].split(" ")
                a = st[1]
                #b = st[3]
                #m = example.index(a)
                #n = example.index(b)
                aa = st[0]
                #bb = st[2]
                ll = []
                rr = []
                ll_tag =[]
                rr_tag =[]
                model= loadGloveModel(file) 
                ll.append(model[aa])
                ll_tag.append(st[3])

                for items in example:
                    temp = items.split(" ")
                    if temp[0] == aa:
                        rr.append(model[temp[2]])
                        
                        rr_tag.append(temp[3])

                left,right = attention(ll,rr)

                for p in range(0,len(rr_tag)):
                    A = left[0][rr[p]]
                    A_ = right[rr[p][0]]
                    q = example.indes(tag_list[p])

                    cp_loss += (A+A_)*(abs(self.logits[i]-self.logits[q]))


        f.close()
        g.close()
        self.loss = self.loss + cp_loss
        return self.loss



    



    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()
        self.add_coupling_loss()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        #for i in words:
            #print(str(i))
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        #print("From predict batch: " + str(words))
        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]
            #print("LABELS: " +str(viterbi_sequences))
            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels,_dummy) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)
            #print(labels)

            #print (fd)
            self.loss = self.loss
            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

        




            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test ):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        out = ''
        
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels, sumit in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)
            
            for desc, lab, lab_pred, length in zip(sumit, labels, labels_pred,
                                             sequence_lengths):
                
                for i, j, k in zip(desc, lab, lab_pred):
                    out = out + str(self.idx_to_word[i[1]]) + " "+str(self.idx_to_tag[j]) + " " + str(self.idx_to_tag[k]) + "\n"
                out = out + "\n"
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        with open("./test_out.txt", 'w') as f:
            f.write(out)
        print ("Precision:" + " " + str(p) + " " + "Recall:" + " " + str(r))
        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        #print(words_raw)
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
