# coding:utf-8
# import os
# import codecs
# import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, GRU, BatchNormalization, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from bert.extract_feature import BertVector
import pickle as pl

"""
将bert预训练模型（chinese_L-12_H-768_A-12）放到当前目录下
基于bert句向量的文本分类：基于Dense的微调
"""
"""GPU设置为按需增长"""
# import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# 指定第一块GPU可用
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# # config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
# sess = tf.Session(config=config)
# KTF.set_session(sess)
############################################
label,content,label_content = pl.load(open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\keras_bert_classification\data\wos_labelcontent_label_content2', 'rb'))

train_label, test_label, train_labcont, test_labcont = train_test_split(label, label_content, test_size=0.2,
                                                                        random_state=42)
wosy2, wosy2_to_id,_, _ = pl.load(open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\keras_bert_classification\data\label2idclean', 'rb'))
##############################################
class BertClassification(object):
    def __init__(self,
                 nb_classes=143,
                 dense_dim=256,
                 max_len=128,
                 batch_size=32,
                 epochs=30,
                 train_corpus_path="data/sent.train",
                 test_corpus_path="data/sent.test",
                 weights_file_path="./model/bertweights_fc.h5"):
        self.nb_classes = nb_classes
        self.dense_dim = dense_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights_file_path = weights_file_path
        self.train_corpus_path = train_corpus_path
        self.test_corpus_path = test_corpus_path

        self.nb_samples = 46985 # 样本数
        self.bert_model = BertVector(pooling_strategy="REDUCE_MEAN", 
                                     max_seq_len=self.max_len,
                                     bert_model_path=r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\keras_bert_classification\uncased_L-12_H-768_A-12",
                                     graph_tmpfile="./data/output/tmp_graph_xxx")

    def text2bert(self, text):
        """ 将文本转换为bert向量  """
        vec = self.bert_model.encode([text])
        return vec["encodes"][0]
    #############################################################################################

    #############################################################################################
    def data_format(self, lines):
        """ 将数据转换为训练格式，输入为列表  """
        X, y = [], []
        for line in lines:
            line = line.strip().split("\t")
            # label = int(line[0])
            label = wosy2_to_id[line[0]]
            content = line[1]
            vec = self.text2bert(content)
            X.append(vec)
            y.append(label)
        X = np.array(X)
        y = np_utils.to_categorical(np.asarray(y), num_classes=self.nb_classes)
        return X, y

    def data_iter(self):
        """ 数据生成器 """
        # fr = codecs.open(self.train_corpus_path, "r", "utf-8")
        # lines = fr.readlines()
        # fr.close()
        # random.shuffle(lines)
        lines = train_labcont
        while True:
            for index in range(0, len(lines), self.batch_size):
                batch_samples = lines[index: index+self.batch_size]
                X, y = self.data_format(batch_samples)
                yield (X, y)

    def data_val(self):
        """ 测试数据 """
        # fr = codecs.open(self.test_corpus_path, "r", "utf-8")
        # lines = fr.readlines()
        # fr.close()
        # random.shuffle(lines)
        lines = test_labcont
        X, y = self.data_format(lines)
        return X,y

    def create_model(self):
        x_in = Input(shape=(768, ))
        # tanh
        x_out = Dense(self.dense_dim, activation="relu")(x_in)
        x_out = BatchNormalization()(x_out)
        x_out = Dense(self.nb_classes, activation="softmax")(x_out)
        model = Model(inputs=x_in, outputs=x_out)
        return model

    def train(self):
        model = self.create_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        model.summary()
        checkpoint = ModelCheckpoint(self.weights_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_acc', patience=2, mode='max')
        x_test, y_test = self.data_val()
        model.fit_generator(self.data_iter(),
                            steps_per_epoch=int(self.nb_samples/self.batch_size)+1,
                            epochs=self.epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            validation_steps=None,
                            callbacks=[checkpoint,early_stopping]
                            )


if __name__ == "__main__":
    bc = BertClassification()
    bc.train()




