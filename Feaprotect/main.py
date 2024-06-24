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
# seed_value = 2
# tf.random.set_seed(seed_value)
# np.random.seed(seed_value)
# random.seed(seed_value)


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



    def train(self) -> None:

        model_v1 = self.experiment.model
        model_v2 = self.experiment.modelv2

        for layer_index in range(len(model_v1.layers)):
            if layer_index == 0 or layer_index == 1:
                continue
            model_v2.layers[layer_index].set_weights(model_v1.layers[layer_index].get_weights())

        if params.need_eg:
            attributions_1_all = eager_ops.expected_gradients_full(inputs=self.v1_train_inputs,
                                                            references=self.v1_train_inputs,
                                                            model=self.experiment.model,
                                                            k=100,
                                                            index_true_class=True,
                                                            labels=self.v2_train_labels)
        else:
            attributions_1_all = None

        train_loss = metrics.Mean()
        l2_norm_loss = metrics.Mean()
        pred_loss_loss = metrics.Mean()
        eg_loss_loss = metrics.Mean()
        val_accuracy = metrics.CategoricalAccuracy()
        best_val_result = float('-inf')
        start_time = time.time()
        train_start_time = time.time()
        index_dataset = np.reshape(range(len(self.v2_train_inputs)), (-1, 1))
        for epoch in range(params.epochs):
            for x, y, batch_indices in tf.data.Dataset.from_tensor_slices((self.v2_train_inputs, self.v2_train_labels, index_dataset)).batch(params.batch_size):
                loss_value, l2_norm, pred_loss, eg_loss = self.train_step(model_v2, self.optimizer, x, y, batch_indices, attributions_1_all)
                train_loss(loss_value)
                l2_norm_loss(l2_norm)
                pred_loss_loss(pred_loss)
                eg_loss_loss(eg_loss)

            val_accuracy(self.v2_val_labels, model_v2(self.v2_val_inputs, training=False))
            eg_cos_sim = self.cal_eg_cos_sim(self.v2_val_inputs, self.v2_val_labels, model_v2)

            if val_accuracy.result() >= best_val_result:
                best_val_result = val_accuracy.result()
                model_v2.save(self.best_model_save_path)
                print('best epoch:', epoch)
            print(
                'Epoch {}, Train Loss: {:.4f},  l2_norm_loss: {:.4f}, pred_loss:{:.4f}, eg_loss: {:.4f} ,Val Accuracy: {:.4f}, ({:.1f} seconds / epoch), cos_sim: {:.4f}.'.format(epoch,
                                                                                                      train_loss.result(),
                                                                                                      l2_norm_loss.result(),
                                                                                                      pred_loss_loss.result(),
                                                                                                      eg_loss_loss.result(),
                                                                                                      val_accuracy.result(),
                                                                                                      time.time() - start_time,
                                                                                                      eg_cos_sim))

            train_loss.reset_states()
            l2_norm_loss.reset_states()
            val_accuracy.reset_states()
            pred_loss_loss.reset_states()
            eg_loss_loss.reset_states()
            start_time = time.time()

        train_end_time = time.time()
        train_execution_time = train_end_time - train_start_time
        print(f"train time: {train_execution_time:.2f} 秒")
        
    def cal_eg_cos_sim(self, test_inputs, test_outputs, model2):
        attributions = eager_ops.expected_gradients_full(inputs=test_inputs,
                                                         references=self.v2_train_inputs,
                                                         model=model2,
                                                         k=100,
                                                         index_true_class=True,
                                                         labels=test_outputs)
        v1_test_inputs = np.delete(test_inputs, self.experiment.dataset['drop_columns_ids'], axis=1)

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
        return cos_sim


    def loss_fn(self, y_true, y_pred, x, model, batch_indices, attributions_1_all):

        pred_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        total_loss = pred_loss
        eg_loss = 0
        l2_norm = 0
        if params.need_eg:
            attributions = eager_ops.expected_gradients_full(inputs=x,
                                                            references=self.v2_train_inputs,
                                                            model=model,
                                                            k=100,
                                                            index_true_class=True,
                                                            labels=y_true)
            
            attributions_1 = attributions_1_all[tf.reshape(batch_indices, shape=[-1])[0].numpy(): tf.reshape(batch_indices, shape=[-1])[-1].numpy()+1]

            mask = tf.constant([True if i not in self.experiment.dataset['drop_columns_ids'] else False for i in range(x.shape[1])])
            attributions_2 = tf.boolean_mask(attributions, mask, axis=1)
            attributions_1 = tf.constant(attributions_1)

            sum_attributions_2 = tf.reduce_sum(attributions_2)
            attributions_2 = tf.divide(attributions_2, sum_attributions_2)
            sum_attributions_1 = tf.reduce_sum(attributions_1)
            attributions_1 = tf.divide(attributions_1, sum_attributions_1)

            cosine_similarity = tf.keras.losses.CosineSimilarity(axis=1)(attributions_1, attributions_2)
            eg_loss = cosine_similarity + 1

            total_loss = params.lamb * total_loss + (1 - params.lamb) * eg_loss

        return total_loss, l2_norm, pred_loss, eg_loss


    def train_step(self, model, optimizer, x, y, batch_indices, attributions_1_all):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss, l2_norm, pred_loss, eg_loss = self.loss_fn(y, y_pred, x, model, batch_indices, attributions_1_all)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, l2_norm, pred_loss, eg_loss

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
    if not params.fix_type == 'ori':
        fm.train()

    fm.evaluate()
