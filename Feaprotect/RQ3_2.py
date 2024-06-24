import os
from tensorflow.keras import metrics
from src.experiment_builder import get_experiment
import sys
import tensorflow as tf
sys.path.append('autoencoder')
from utils.logger import logger
logger = logger(__name__)
import warnings
warnings.filterwarnings("ignore")
import utils.struct_util as StructUtil
import random
import numpy as np
import utils.expect_grad_ops_util as eager_ops
random.seed(42)
np.random.seed(42)
from scipy import spatial
import time
import os,sys
from tensorflow.keras.losses import MeanSquaredError
os.chdir(sys.path[0])
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def acc_nf(output_new, output_old, target, topk=(1,)):
    maxk = max(topk)
    batch_size = tf.shape(target)[0]

    _, pred_new = tf.nn.top_k(output_new, k=maxk)
    _, pred_old = tf.nn.top_k(output_old, k=maxk)
    pred_new = tf.transpose(pred_new)
    pred_old = tf.transpose(pred_old)
    target_expanded = tf.expand_dims(target, axis=0)
    target_expanded = tf.tile(target_expanded, [maxk, 1])
    correct_new = tf.equal(tf.cast(pred_new, tf.int64), target_expanded)
    correct_old = tf.equal(tf.cast(pred_old, tf.int64), target_expanded)

    nf = tf.math.logical_not(correct_new) & correct_old
    acc = []
    nfr = []

    for k in topk:
        correct_k = tf.reduce_sum(tf.cast(tf.reshape(correct_new[:k], [-1]), tf.float32))
        acc.append(correct_k * 100.0 / tf.cast(batch_size, tf.float32))

        nfr_k = tf.reduce_sum(tf.cast(tf.reshape(nf[:k], [-1]), tf.float32))
        nfr.append(nfr_k * 100.0 / tf.cast(batch_size, tf.float32))

    return acc, nfr

def temperature_scale(logits, T):
    temperature = tf.expand_dims(T, axis=1)
    temperature = tf.tile(temperature, [logits.shape[0], logits.shape[1]])
    return logits / temperature

class ensemble_model(tf.keras.Model):

    def __init__(self, model1, model2, opt):
        super(ensemble_model, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.ensemble = opt.ensemble
        if opt.ensemble == 'weight':
            self.k = opt.weight

    def call(self, v1_data, v2_data):
        output1 = self.model1(v1_data)
        output2 = self.model2(v2_data)
        if 'average' in self.ensemble:
            old_confs = tf.keras.activations.softmax(output1, axis=-1)
            new_confs = tf.keras.activations.softmax(output2, axis=-1)
            output = (new_confs + old_confs) / 2
        elif 'maximum' in self.ensemble:
            old_confs = tf.nn.softmax(output1, axis=-1)
            new_confs = tf.nn.softmax(output2, axis=-1)
            p = tf.reduce_max(old_confs, axis=1)
            q = tf.reduce_max(new_confs, axis=1)
            output = tf.identity(output2)
            ind = tf.where(p > q)
            output = tf.tensor_scatter_nd_update(output, ind, tf.gather(output1, ind))
        elif "perturb" in self.ensemble:
            loops = 10
            pred1 = []
            pred2 = []
            for _ in range(loops):
                pert = tf.random.normal(v1_data.shape)
                pred1.append(tf.nn.softmax(self.model1(v1_data + 0.01 * pert), axis=-1))
                pert = tf.random.normal(v2_data.shape)
                pred2.append(tf.nn.softmax(self.model2(v2_data + 0.01 * pert), axis=-1))
            var_pred1 = tf.zeros(output1.shape)
            var_pred2 = tf.zeros(output2.shape)
            mean_pred1 = tf.zeros(output1.shape)
            mean_pred2 = tf.zeros(output2.shape)
            for _ in range(loops):
                mean_pred1 += pred1[_]
                mean_pred2 += pred2[_]
                var_pred1 += tf.square(pred1[_])
                var_pred2 += tf.square(pred2[_])
            var_pred1 = (var_pred1 / loops) - tf.square(mean_pred1 / loops)
            var_pred2 = (var_pred2 / loops) - tf.square(mean_pred2 / loops)
            weight1 = tf.clip_by_value(tf.reduce_sum(var_pred1, axis=-1), clip_value_min=1e-3, clip_value_max=1-1e-3)
            weight2 = tf.clip_by_value(tf.reduce_sum(var_pred2, axis=-1), clip_value_min=1e-3, clip_value_max=1-1e-3)
            w1 = tf.expand_dims((weight2 / (2 * (weight1 + weight2)) + 0.0), axis=-1)
            w2 = tf.expand_dims((weight1 / (2 * (weight1 + weight2)) + 0.5), axis=-1)
            output = tf.multiply(mean_pred1 / loops, w1) + tf.multiply(mean_pred2 / loops, w2)
        
        elif "dropout" in self.ensemble:
            loops = 10
            model1_with_dropout = tf.keras.Sequential([
                self.model1,
                tf.keras.layers.Dropout(rate=0.2)
            ])
            model2_with_dropout = tf.keras.Sequential([
                self.model2,
                tf.keras.layers.Dropout(rate=0.2)
            ])

            pred1 = []
            pred2 = []
            for _ in range(loops):
                pred1.append(tf.nn.softmax(model1_with_dropout(v1_data, training=False), axis=-1))
                pred2.append(tf.nn.softmax(model2_with_dropout(v2_data, training=False), axis=-1))
            
            var_pred1 = tf.zeros(output1.shape)
            var_pred2 = tf.zeros(output2.shape)
            mean_pred1 = tf.zeros(output1.shape)
            mean_pred2 = tf.zeros(output2.shape)
            for _ in range(loops):
                mean_pred1 += pred1[_]
                mean_pred2 += pred2[_]
                var_pred1 += tf.square(pred1[_])
                var_pred2 += tf.square(pred2[_])
            
            var_pred1 = (var_pred1 / loops) - tf.square(mean_pred1 / loops)
            var_pred2 = (var_pred2 / loops) - tf.square(mean_pred2 / loops)

            weight1 = tf.reduce_sum(var_pred1, axis=-1)
            weight2 = tf.reduce_sum(var_pred2, axis=-1)

            weight1 = tf.clip_by_value(tf.reduce_sum(var_pred1, axis=-1), clip_value_min=1e-3, clip_value_max=1-1e-3)
            weight2 = tf.clip_by_value(tf.reduce_sum(var_pred2, axis=-1), clip_value_min=1e-3, clip_value_max=1-1e-3)

            w1 = tf.expand_dims((weight2 / (2 * (weight1 + weight2))) + 0.0, -1)
            w2 = tf.expand_dims((weight1 / (2 * (weight1 + weight2))) + 0.5, -1)

            output = tf.multiply(mean_pred1 / loops, w1) + tf.multiply(mean_pred2 / loops, w2)
        
        elif 'scaling_unlabel' in self.ensemble:
            old_confs = tf.nn.softmax(temperature_scale(output1, self.T1), axis=-1)
            new_confs = tf.nn.softmax(temperature_scale(output2, self.T2), axis=-1)
            output = old_confs * self.k + new_confs * (1 - self.k)
        return output
    

    def scaling(self, data_loader, opt):
        old_model = self.model1
        new_model = self.model2
        old_model.compile()
        new_model.compile()

        old_logit_list = []
        new_logit_list = []
        labels_list = []
        for o_x, n_x, label in data_loader:
            old_logit = old_model(o_x)
            new_logit = new_model(n_x)
            old_logit_list.append(old_logit)
            new_logit_list.append(new_logit)
            labels_list.append(tf.argmax(label, axis=1))

        old_logit_list = tf.concat(old_logit_list, axis=0)
        new_logit_list = tf.concat(new_logit_list, axis=0)
        labels = tf.concat(labels_list, axis=0)

        T1 = tf.Variable([1.0], dtype=tf.float32)
        p_ = tf.nn.softmax(temperature_scale(old_logit_list, T1), axis=-1)
        q_ = tf.nn.softmax(new_logit_list, axis=-1)
        p, old_pred = tf.reduce_max(p_, axis=1), tf.argmax(p_, axis=1)
        q, new_pred = tf.reduce_max(q_, axis=1), tf.argmax(q_, axis=1)

        def eval(k):
            output = p_ * k + q_ * (1 - k)
            conf, pred = tf.reduce_max(output, axis=1), tf.argmax(output, axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))
            nfr = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(old_pred, labels), tf.not_equal(pred, labels)), tf.float32))
            return acc, nfr

        acc = 0.0
        interval = np.arange(0.0, 1.001, 0.001)
        acc_low = tf.reduce_mean(tf.cast(tf.equal(new_pred, labels), tf.float32))
        best_nfr = 1.0
        best_acc = 0.0
        best_k = 0.0
        acc_list = []
        nfr_list = []

        for i in range(len(interval)):
            k = interval[i]
            acc, nfr = eval(k)
            acc_list.append(acc.numpy())
            nfr_list.append(nfr.numpy())
            np.savetxt('curve.csv', [acc_list, nfr_list])
            if acc >= acc_low and nfr < best_nfr:
                best_k = k
                best_nfr = nfr
                best_acc = acc

        self.k = tf.constant([best_k], dtype=tf.float32)
        T1 = tf.Variable([1.0], dtype=tf.float32)
        T2 = tf.Variable([1.0], dtype=tf.float32)
        self.T1 = T1
        self.T2 = T2
        print(self.k)
        return self.T1, self.T2, self.k
    
    def scaling_unlabel(self, data_loader, opt):
        old_model = self.model1
        new_model = self.model2
        old_logit_list = []
        new_logit_list = []
        for o_x, n_x, label in data_loader:
            old_logit = old_model(o_x)
            new_logit = new_model(n_x)
            old_logit_list.append(old_logit)
            new_logit_list.append(new_logit)

        old_logit_list = tf.concat(old_logit_list, axis=0).numpy()
        new_logit_list = tf.concat(new_logit_list, axis=0).numpy()

        def solveT(old_logit_list, new_logit_list, init_T=1.0, reg=0.0):
            T1 = tf.Variable([init_T], dtype=tf.float32)
            optimizer = tf.optimizers.SGD(learning_rate=1.0)
            mse_loss = MeanSquaredError()
            
            @tf.function
            def eval():
                with tf.GradientTape() as tape:
                    p = tf.nn.softmax(temperature_scale(old_logit_list, T1), axis=-1)
                    q = tf.nn.softmax(new_logit_list, axis=-1)
                    loss = mse_loss(p, q) + reg * tf.square(T1)
                gradients = tape.gradient(loss, [T1])
                optimizer.apply_gradients(zip(gradients, [T1]))
                return loss
            
            for _ in range(100):
                eval()
            
            return T1

        init_list = [1.0, 2.0, 3.0]
        best_loss = 1e6
        for T in init_list:
            mse_loss = MeanSquaredError()
            T1 = solveT(old_logit_list, new_logit_list, init_T=T, reg=5e-4)
            with tf.GradientTape():
                p = tf.nn.softmax(temperature_scale(old_logit_list, T1), axis=-1)
                q = tf.nn.softmax(new_logit_list, axis=-1)
                loss = mse_loss(p, q)
            if loss < best_loss and T1 > 0:
                best_T = T1
                best_loss = loss
            print('T: {}, loss: {}'.format(T1.numpy(), loss.numpy()))

        if best_T < 0.8:
            init_list = [1.0, 2.0, 3.0]
            best_loss = 1e6
            for T in init_list:
                mse_loss = MeanSquaredError()
                T1 = solveT(old_logit_list, new_logit_list, init_T=T, reg=0.0)
                with tf.GradientTape():
                    p = tf.nn.softmax(temperature_scale(old_logit_list, T1), axis=-1)
                    q = tf.nn.softmax(new_logit_list, axis=-1)
                    loss = mse_loss(p, q)
                if loss < best_loss and T1 > 0:
                    best_T = T1
                    best_loss = loss
                print('T: {}, loss: {}'.format(T1.numpy(), loss.numpy()))

        print('set T by {}'.format(best_T.numpy()))
        self.T1 = best_T
        self.T2 = tf.Variable([1.0], dtype=tf.float32)
        self.k = tf.constant([0.5], dtype=tf.float32)

class fixmodel:
    def __init__(self, params) -> None:
        self.params = params

        self.experiment = get_experiment(self.params)
        self.modelv2_acc, self.modelv1_acc = self.get_ori_acc()
        print('modelv1_acc: {}'.format(self.modelv1_acc))
        print('modelv2_acc: {}'.format(self.modelv2_acc))

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.prepare_data()
        self.model2 = tf.keras.models.load_model(os.path.join('models', params.dataset.lower(), params.dataset_col, params.fix_type, 'v2_weights.hdf5'))
        self.model1 = tf.keras.models.load_model(os.path.join('models', params.dataset.lower(), params.dataset_col, 'v1_weights.hdf5'))
        self.model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        layer_name = 'beforesoftmax'
        self.beforesoftmax_model1 = tf.keras.Model(inputs=self.model1.input, outputs=self.model1.get_layer(layer_name).output)
        self.beforesoftmax_model2 = tf.keras.Model(inputs=self.model2.input, outputs=self.model2.get_layer(layer_name).output)

    def get_ori_model_v2_acc(self, inputs, labels, experiment) -> float:
        ori_model_v2 = tf.keras.models.load_model(
            os.path.join('models', params.dataset.lower(), params.dataset_col, 'v2_weights.hdf5'))
        ori_model_v2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        modelv2_acc = ori_model_v2.evaluate(inputs, labels)[1]
        return modelv2_acc

    def get_ori_model_v1_acc(self, inputs, labels, experiment) -> float:
        ori_model_v2 = tf.keras.models.load_model(
            os.path.join('models', params.dataset.lower(), params.dataset_col, 'v1_weights.hdf5'))
        ori_model_v2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        inputs = np.delete(inputs, experiment.dataset['drop_columns_ids'], axis=1)
        modelv2_acc = ori_model_v2.evaluate(inputs, labels)[1]
        return modelv2_acc

    def get_ori_acc(self) -> tuple:
        _params = get_params()
        _params.fix_type = 'ori'
        experiment = get_experiment(_params)
        v2_test_inputs = StructUtil.preprocess_data(experiment.dataset['test_inputs'], experiment.dataset['scaler2'])
        v2_test_labels = experiment.dataset['test_outputs']
        modelv2_acc = self.get_ori_model_v2_acc(v2_test_inputs.copy(), v2_test_labels.copy(), experiment)
        modelv1_acc = self.get_ori_model_v1_acc(v2_test_inputs.copy(), v2_test_labels.copy(), experiment)
        logger.info('modelv2_acc: {}, modelv1_acc: {}'.format(modelv2_acc, modelv1_acc))
        del experiment
        del v2_test_inputs
        del v2_test_labels
        return modelv2_acc, modelv1_acc


    def prepare_data(self) -> None:
        self.v2_train_inputs = StructUtil.preprocess_data(self.experiment.dataset['train_inputs'], self.experiment.dataset['scaler2'])
        self.v2_train_inputs = np.array(self.v2_train_inputs, dtype=np.float32)
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


        self.v1_train_inputs = np.delete(self.v2_train_inputs, self.experiment.dataset['drop_columns_ids'], axis=1)
        self.v1_train_labels = self.v2_train_labels

        self.v1_test_inputs_1 = np.delete(self.v2_test_inputs_1, self.experiment.dataset['drop_columns_ids'], axis=1)
        self.v1_test_labels_1 = self.v2_test_labels_1

        self.v1_val_inputs = np.delete(self.v2_val_inputs, self.experiment.dataset['drop_columns_ids'], axis=1)
        self.v1_val_labels  = self.v2_val_labels 


    def create_ensemble_model(self, v1_data, v2_data, targets, model1, model2) -> None:

        val_loader = tf.data.Dataset.from_tensor_slices((self.v1_val_inputs, self.v2_val_inputs, self.v2_val_labels)).batch(params.batch_size)
        conbine_model = ensemble_model(model1=model1, model2=model2, opt=self.params)
        if 'average' in self.params.ensemble:
            self.params.softmax = False

        elif 'scaling_unlabel' in self.params.ensemble:
            conbine_model.scaling_unlabel(val_loader, self.params)
            self.params.softmax = False
        elif 'dropout' in self.params.ensemble:
            self.params.softmax = False
        elif 'perturb' in self.params.ensemble:
            self.params.softmax = False
        self.conbine_model = conbine_model
        return conbine_model

    def get_test_acc(self, v1_test_inputs, v2_test_inputs, test_outputs, model1, combine_model):

        test_accuracy = metrics.CategoricalAccuracy()
        v2_scores = test_accuracy(test_outputs, combine_model(v1_test_inputs, v2_test_inputs)).numpy()


        modelv2_y_pred = combine_model(v1_test_inputs, v2_test_inputs)
        modelv1_y_pred = model1.predict(v1_test_inputs)
        NF = 0
        new_model_nf_index_list = []
        for idx, (v1_p, v2_p, t) in enumerate(zip(modelv1_y_pred, modelv2_y_pred, test_outputs)):
            if np.argmax(v1_p) == np.argmax(t) and np.argmax(v2_p) != np.argmax(t):
                NF += 1
                new_model_nf_index_list.append(idx)
        NFR = NF / len(v1_test_inputs)


        test_accuracy = metrics.CategoricalAccuracy()
        v1_er = 1 - test_accuracy(test_outputs, model1(v1_test_inputs, training=False)).numpy()

        test_accuracy.reset_states()
        v2_er = 1 - test_accuracy(test_outputs, combine_model(v1_test_inputs, v2_test_inputs)).numpy()
        NFR_REL = NFR / ((1 - v1_er) * v2_er)


        print("fix model acc: {:.4f}".format(v2_scores))
        print("NF: {}".format(NF))
        print("NFR: {:.4f}".format(NFR))
        print("NFR_REL: {:.4f}".format(NFR_REL))


    def calculate_intervals(self, lst, num_bins=10):

        min_val = 0.5
        max_val = 1        
        interval_length = (max_val - min_val) / num_bins
        
        intervals = {i: {'min': min_val + i * interval_length, 
                        'max': min_val + (i+1) * interval_length,
                        'total': 0,
                        'count': 0} 
                        for i in range(num_bins)}
        intervals[num_bins-1]['max'] = max_val + 0.1   

        for item in lst:
            value = item[0]
            flag = item[1]

            for i in range(10):
                if intervals[i]['min'] <= value < intervals[i]['max']:
                    intervals[i]['total'] += 1
            
            for i in range(10):
                if intervals[i]['min'] <= value < intervals[i]['max']:
                    intervals[i]['count'] += flag
        
        return intervals


    def evaluate(self, combine_model) -> None:
        if self.params.rf:
            rf_data_input = self.experiment.dataset['new_rf_data_input']
            rf_data_output = self.experiment.dataset['new_rf_data_output']
            old_model2_rf_data_input  = rf_data_input.copy()
            if 'fea_sel' in self.params.fix_type and len(self.experiment.dataset['new_col_conflict']) != 0:
                rf_data_input = np.delete(rf_data_input, self.experiment.dataset['new_col_conflict'], axis=1)

            rf_data_input = StructUtil.preprocess_data(rf_data_input, self.experiment.dataset['scaler2'])
            rf_data_output = np.array(rf_data_output)
            v1_rf_data_input = np.delete(rf_data_input, self.experiment.dataset['drop_columns_ids'], axis=1)


            rf_accuracy = metrics.CategoricalAccuracy()
            rf_data_output = tf.keras.utils.to_categorical(rf_data_output, num_classes=params.num_classes)
            rf_scores = rf_accuracy(rf_data_output, combine_model(v1_rf_data_input, rf_data_input)).numpy()
            print("rf_scores: {:.4f}".format(rf_scores))


            old_model2 = tf.keras.models.load_model(os.path.join('models', params.dataset.lower(), params.dataset_col, 'v2_weights.hdf5'))
            old_model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            rf_data = np.load(os.path.join('dataset', params.dataset.lower(), params.dataset_col, 'rf_data', params.rf_name), allow_pickle=True)
            old_model2_rf_data_input = [item.input for item in rf_data]
            if 'fea_sel' in self.params.fix_type:
                old_model2_rf_data_input = StructUtil.preprocess_data(old_model2_rf_data_input, self.experiment.dataset['old_scaler2'])
            else:
                old_model2_rf_data_input = StructUtil.preprocess_data(old_model2_rf_data_input, self.experiment.dataset['scaler2'])
            old_model2_y_pred = old_model2.predict(old_model2_rf_data_input)
            confs_o = tf.reduce_max(old_model2_y_pred, axis=1).numpy()

            new_model2_n_pred = combine_model(v1_rf_data_input, rf_data_input)
            new_model2_n_p = [1 if np.argmax(y_p) == np.argmax(t) else 0 for y_p, t in zip(new_model2_n_pred, rf_data_output)]
            conbine_result = []
            for conf, newmodel_p in zip(confs_o, new_model2_n_p):
                conbine_result.append((conf, newmodel_p))

            num_bins = 10
            result = self.calculate_intervals(conbine_result, num_bins=num_bins)


        self.get_test_acc(self.v1_test_inputs_1, self.v2_test_inputs_1, self.v2_test_labels_1, self.model1, combine_model)

    
    def rho_estimate(self, v1_data, v2_data, targets, model1, model2):
        old_logits = []
        new_logits = []
        labels = []
        for o_x, n_x, y in tf.data.Dataset.from_tensor_slices((v1_data, v2_data, targets)).batch(params.batch_size):
            y = tf.argmax(y, axis=1)
            old_output = model1(o_x, training=False)
            new_output = model2(n_x, training=False)
            old_logits.append(old_output)
            new_logits.append(new_output)
            labels.append(y)
        
        old_logits = tf.concat(old_logits, 0)
        new_logits = tf.concat(new_logits, 0)
        labels = tf.concat(labels, 0)
        if self.params.softmax == True:
            old_confs = tf.keras.activations.softmax(old_logits, axis=-1)
            new_confs = tf.keras.activations.softmax(new_logits, axis=-1)
        else:
            old_confs = old_logits
            new_confs = new_logits
        from scipy import stats
        import numpy as np
        corrs = []
        for k in range(self.params.num_classes):
            rho = stats.pearsonr(old_confs[:, k], new_confs[:, k])
            corrs.append(rho)
        
        mse_corr = tf.reduce_mean(tf.reduce_sum(tf.square(new_confs - old_confs), axis=1))
        print('Correlation of old and new model is {:.4f}, mse is {:.4f}'.format(np.mean(corrs), mse_corr))
        return 
    

    def nfr_validate(self, v1_data, v2_data, targets, model1, model2):
        batch_time = AverageMeter()
        losses = AverageMeter()
        nfr_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        nfr_top1 = AverageMeter()
        nfr_top5 = AverageMeter()

        end = time.time()
        for i, (o_x, n_x, y) in enumerate(tf.data.Dataset.from_tensor_slices((v1_data, v2_data, targets)).batch(params.batch_size)):
            y = tf.argmax(y, axis=1)
            old_output = model1(o_x, training=False)
            new_output = model2(n_x, training=False)
            softmaxes_o = tf.keras.activations.softmax(old_output, axis=-1)
            softmaxes_o = softmaxes_o * tf.cast(tf.one_hot(y, depth=params.num_classes), dtype=tf.float32)
            if self.params.softmax == True:
                softmaxes_n = tf.keras.activations.softmax(new_output, axis=-1)
            else:
                softmaxes_n = new_output
            softmaxes_n = softmaxes_n * tf.cast(tf.one_hot(y, depth=params.num_classes), dtype=tf.float32)
            confs_o, _ = tf.reduce_max(softmaxes_o, axis=1), tf.argmax(softmaxes_o, axis=1)
            confs_n, _ = tf.reduce_max(softmaxes_n, axis=1), tf.argmax(softmaxes_n, axis=1)
            nfr_loss = -tf.reduce_mean(tf.math.log(confs_n + (1 - confs_o) * (1 - confs_n)))
            loss = -tf.reduce_mean(tf.math.log(confs_n))
            if self.params.num_classes <= 5:
                acc1, nfr1 = acc_nf(new_output, old_output, y, topk=(1,))
                top1.update(acc1[0].numpy(), y.shape[0])
                top5.update(1.0, y.shape[0])
                nfr_top1.update(nfr1[0].numpy(), y.shape[0])
                nfr_top5.update(0.0, y.shape[0])
            else:
                acc, nfr = acc_nf(new_output, old_output, y, topk=(1, 5))
                acc1, acc5 = acc[0], acc[1]
                nfr1, nfr5 = nfr[0], nfr[1]
                top1.update(acc1[0], y.shape[0])
                top5.update(acc5[0], y.shape[0])
                nfr_top1.update(nfr1[0], y.shape[0])
                nfr_top5.update(nfr5[0], y.shape[0])     
            losses.update(loss.numpy(), y.shape[0])
            nfr_losses.update(nfr_loss, y.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} NFR@1 {nfr_top1.avg:.3f} NFR@5 {nfr_top5.avg:.3f} ce_loss {loss.avg:.3f} nfr_loss {nfr_loss.avg:.3f}'
              .format(top1=top1, top5=top5, nfr_top1=nfr_top1, nfr_top5=nfr_top5, loss=losses, nfr_loss=nfr_losses))



def get_params():
    import argparse
    parser = argparse.ArgumentParser(description="Experiments Script For DeepReFuzz")
    parser.add_argument("--dataset", type=str, default="mobile_price")
    parser.add_argument("--dataset_col", type=str,
                        default="fc-m_dep-int_memory-mobile_wt-pc-px_height")
    parser.add_argument("--model", type=str, default="mobile_price")
    parser.add_argument("--update_col_num", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--fix_type", type=str, default='eg_discrt_fea_sel_1')
    parser.add_argument("--ensemble", type=str, default='scaling_unlabel')
    parser.add_argument("--softmax", type=bool, default=False)
    parser.add_argument("--rf", type=bool, default=True)
    parser.add_argument("--rf_name", type=str, default="rf.npy")
    params = parser.parse_args()

    return params


if __name__ == '__main__':
    params = get_params()
    params.epochs = 100
    params.batch_size = 32
    fm = fixmodel(params)
    
    conbine_model = fm.create_ensemble_model(fm.v1_test_inputs_1, fm.v2_test_inputs_1, fm.v2_test_labels_1, fm.beforesoftmax_model1, fm.beforesoftmax_model2)

    fm.evaluate(conbine_model)