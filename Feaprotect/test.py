import os
from tensorflow.keras import metrics
from src.experiment_builder import get_experiment
from utils.param_util import get_params
import sys
import tensorflow as tf
from utils.logger import logger
logger = logger(__name__)
import warnings
warnings.filterwarnings("ignore")
import utils.struct_util as StructUtil
import numpy as np
import utils.expect_grad_ops_util as eager_ops
from scipy import spatial
import time
import os,sys
os.chdir(sys.path[0])
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class fixmodel:
    def __init__(self, params) -> None:
        self.params = params
        self.experiment = get_experiment(self.params)
        learning_rate_dict = {
            "mobile_price": 0.001,
            "fetal_health": 0.001,
            "diabetes": 0.001,
            "customerchurn": 0.001,
            "onlineshoppers": 0.001,
            "musicgenres": 0.001,
            "hand_gesture": 0.001,
            "bean": 0.001,
            "patient": 0.001,
            "climate": 0.001
        }
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_dict[self.experiment.dataset['dataset_name']])
        self.prepare_data()
        self.best_model_save_path = os.path.join('models', params.dataset.lower(), params.dataset_col, params.fix_type,
                                                 'v2_weights.hdf5')
        if not os.path.exists(os.path.join('models', params.dataset.lower(), params.dataset_col, params.fix_type)):
            os.makedirs(os.path.join('models', params.dataset.lower(), params.dataset_col, params.fix_type))

    def prepare_data(self) -> None:
        self.v2_train_inputs = StructUtil.preprocess_data(self.experiment.dataset['train_inputs'], self.experiment.dataset['scaler2'])
        self.v2_train_inputs = np.array(self.v2_train_inputs, dtype=np.float32)
        self.v1_train_inputs = np.delete(self.v2_train_inputs, self.experiment.dataset['drop_columns_ids'], axis=1)
        v2_train_labels = self.experiment.dataset['train_outputs']
        self.v2_train_labels = tf.keras.utils.to_categorical(v2_train_labels, num_classes=params.num_classes)


        self.v2_test_inputs_1 = StructUtil.preprocess_data(self.experiment.dataset['test_inputs'], self.experiment.dataset['scaler2'])
        self.v2_test_inputs_1 = np.array(self.v2_test_inputs_1, dtype=np.float32)
        v2_test_labels_1 = self.experiment.dataset['test_outputs']
        self.v2_test_labels_1 = tf.keras.utils.to_categorical(v2_test_labels_1, num_classes=params.num_classes)


        self.v2_val_inputs = StructUtil.preprocess_data(self.experiment.dataset['val_inputs'], self.experiment.dataset['scaler2'])
        self.v2_val_inputs = np.array(self.v2_val_inputs, dtype=np.float32)
        v2_val_labels = self.experiment.dataset['val_outputs']
        self.v2_val_labels = tf.keras.utils.to_categorical(v2_val_labels, num_classes=params.num_classes)


    def get_test_acc(self, test_inputs, test_outputs, model2):

        test_accuracy = metrics.CategoricalAccuracy()
        v2_scores = test_accuracy(test_outputs, model2(test_inputs, training=False)).numpy()

        v1_test_inputs = np.delete(test_inputs, self.experiment.dataset['drop_columns_ids'], axis=1)
        model1 = self.experiment.model

        attributions = eager_ops.expected_gradients_full(inputs=test_inputs,
                                                         references=self.v2_train_inputs,
                                                         model=model2,
                                                         k=100,
                                                         index_true_class=True,
                                                         labels=test_outputs)

        attributions_1 = eager_ops.expected_gradients_full(inputs=v1_test_inputs,
                                                           references=self.v1_train_inputs,
                                                           model=self.experiment.model,
                                                           k=100,
                                                           index_true_class=True,
                                                           labels=test_outputs)



        attributions_2 = np.delete(attributions, self.experiment.dataset['drop_columns_ids'], axis=1)
        attributions_1 = attributions_1.numpy()

        cos_sim_l = []
        for a1, a2 in zip(attributions_1, attributions_2):
            cos_sim = 1 - spatial.distance.cosine(a1, a2)
            cos_sim_l.append(cos_sim)

        cos_sim = np.mean(cos_sim_l)


        modelv2_y_pred = model2.predict(test_inputs)
        modelv1_y_pred = model1.predict(v1_test_inputs)
        NF = 0
        new_model_nf_index_list = []
        for idx, (v1_p, v2_p, t) in enumerate(zip(modelv1_y_pred, modelv2_y_pred, test_outputs)):
            if np.argmax(v1_p) == np.argmax(t) and np.argmax(v2_p) != np.argmax(t):
                NF += 1
                new_model_nf_index_list.append(idx)
        NFR = NF / len(test_inputs)


        test_accuracy = metrics.CategoricalAccuracy()
        v1_er = 1 - test_accuracy(test_outputs, model1(v1_test_inputs, training=False)).numpy()

        test_accuracy.reset_states()
        v2_er = 1 - test_accuracy(test_outputs, model2(test_inputs, training=False)).numpy()

        NFR_REL = NFR / ((1 - v1_er) * v2_er)

        print("fix model acc: {:.4f}".format(v2_scores))
        print("cos_sim: {:.4f}".format(cos_sim))
        print("NF: {}".format(NF))
        print("NFR: {:.4f}".format(NFR))
        print("NFR_REL: {:.4f}".format(NFR_REL))


    def evaluate(self):
        model1 = self.experiment.model
        if params.fix_type == 'ori':
            model2 = tf.keras.models.load_model(os.path.join('models', params.dataset.lower(), params.dataset_col, 'v2_weights.hdf5'))
        else:
            model2 = tf.keras.models.load_model(self.best_model_save_path)
        model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        if self.params.rf:
            rf_data_input = self.experiment.dataset['new_rf_data_input']
            rf_data_output = self.experiment.dataset['new_rf_data_output']

            if self.params.need_fea_Sel and len(self.experiment.dataset['new_col_conflict']) != 0:
                rf_data_input = np.delete(rf_data_input, self.experiment.dataset['new_col_conflict'], axis=1)

            rf_data_input = StructUtil.preprocess_data(rf_data_input, self.experiment.dataset['scaler2'])
            rf_data_output = np.array(rf_data_output)


            rf_accuracy = metrics.CategoricalAccuracy()
            rf_data_output = tf.keras.utils.to_categorical(rf_data_output, num_classes=params.num_classes)
            rf_scores = rf_accuracy(rf_data_output, model2(rf_data_input, training=False)).numpy()
            print("rf_scores {}".format(rf_scores))

        self.get_test_acc(self.v2_test_inputs_1, self.v2_test_labels_1, model2)


import time
if __name__ == '__main__':
    params = get_params()
    params.epochs = 100
    params.batch_size = 64

    fm = fixmodel(params)

    fm.evaluate()
