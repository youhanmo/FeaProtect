import os
import random
import time

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from utils.logger import logger
logger = logger(__name__)
import json
import time
from tensorflow.keras import layers
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
class Experiment:
    pass


def get_experiment(params):
    experiment = Experiment()
    experiment.dataset = _get_dataset(params, experiment)
    experiment.model = _get_model(params, experiment)
    experiment.modelv2 = _get_model_v2(params, experiment)
    return experiment


def _get_dataset(params, experiment):
    label_name = {
        "mobile_price": "price_range",
        "fetal_health": "fetal_health",
        "foreast": "BC",
        "diabetes": "Outcome",
        "customerchurn": "Churn",
        "onlineshoppers": "Revenue",
        "glass": "Type",
        "musicgenres": 'label',
        "hand_gesture": 'BM',
        "bean": 'Class',
        "patient": 'SOURCE',
        "climate": "ddd_car"
    }
    v1_dataset_path = os.path.join('dataset', params.dataset.lower(), params.dataset_col, 'v1')
    v2_dataset_path = os.path.join('dataset', params.dataset.lower(), params.dataset_col, 'v2')

    train_inputs = np.load(os.path.join(v2_dataset_path, 'X_train.npy'))
    train_labels = np.load(os.path.join(v2_dataset_path, 'y_train.npy'))
    val_inputs = np.load(os.path.join(v2_dataset_path, 'X_val.npy'))
    val_labels = np.load(os.path.join(v2_dataset_path, 'y_val.npy'))
    test_inputs = np.load(os.path.join(v2_dataset_path, 'X_test_1.npy'))
    test_labels = np.load(os.path.join(v2_dataset_path, 'y_test_1.npy'))
    test_inputs_2 = np.load(os.path.join(v2_dataset_path, 'X_test_2.npy'))
    test_labels_2 = np.load(os.path.join(v2_dataset_path, 'y_test_2.npy'))
    test_inputs = np.concatenate([test_inputs, test_inputs_2], axis=0)
    test_labels = np.concatenate([test_labels, test_labels_2], axis=0)
    sc1 = joblib.load(os.path.join(v1_dataset_path, 'sc_v1.pkl'))
    sc2 = joblib.load(os.path.join(v2_dataset_path, 'sc_v2.pkl'))
    drop_columns = params.dataset_col.split('-')
    train_df = pd.read_csv(os.path.join('dataset', params.dataset.lower(), 'train.csv'))
    v1_columns = [col for col in train_df.columns if col not in drop_columns and col != label_name[params.dataset]]
    v2_columns = [col for col in train_df.columns if col != label_name[params.dataset]]
    drop_columns_ids = [v2_columns.index(col) for col in drop_columns]
    dataset_name = params.dataset.lower()

    logger.info(f'load dataset {params.dataset} successfully, test_inputs shape: {test_inputs.shape}')

    return {
        "train_inputs": train_inputs,
        "train_outputs": train_labels,
        "test_inputs": test_inputs,
        "test_outputs": test_labels,
        "val_inputs": val_inputs,
        "val_outputs": val_labels,
        "scaler1": sc1,
        "scaler2": sc2,
        "drop_columns": drop_columns,
        "drop_columns_ids": drop_columns_ids,
        "v1_columns": v1_columns,
        "v2_columns": v2_columns,
        "train_df": train_df,
        "dataset_name": dataset_name,
        "v2_dataset_path": v2_dataset_path
    }


def _get_model_v2(params, experiment):
    import os
    if not os.path.exists(os.path.join(experiment.dataset['v2_dataset_path'], params.fix_type)):
        os.makedirs(os.path.join(experiment.dataset['v2_dataset_path'], params.fix_type))
    fea_data_path = os.path.join(experiment.dataset['v2_dataset_path'], params.fix_type)
    import minepy as minepy
    drop_col_idx_2_d = {}
    drop_col_idx_2_bin = {}

    experiment.dataset['old_scaler2'] = experiment.dataset['scaler2']
    combined_data = np.concatenate([experiment.dataset['train_inputs'], experiment.dataset['test_inputs'], experiment.dataset['val_inputs']])
    combined_series_Y = np.concatenate([experiment.dataset['train_outputs'], experiment.dataset['test_outputs'], experiment.dataset['val_outputs']])
    v1_inputs = np.delete(combined_data, experiment.dataset['drop_columns_ids'], axis=1)

    if params.rf:
        if os.path.exists(os.path.join('dataset', params.dataset.lower(), params.dataset_col, 'rf_data', params.rf_name)):
            rf_data = np.load(os.path.join('dataset', params.dataset.lower(), params.dataset_col, 'rf_data', params.rf_name), allow_pickle=True)
            rf_data_input = [item.input for item in rf_data]
            experiment.dataset['rf_data_input'] = np.array(rf_data_input)
            experiment.dataset['rf_data_output'] = [item.label for item in rf_data]
        rf_data = np.load(
            os.path.join('dataset', params.dataset.lower(), params.dataset_col, 'rf_data', 'rf.npy'),
            allow_pickle=True)
        rf_data_input = [item.input for item in rf_data]
        experiment.dataset['new_rf_data_input'] = np.array(rf_data_input)
        experiment.dataset['new_rf_data_output'] = [item.label for item in rf_data]
    if params.need_fea_Sel:
        if not os.path.exists(os.path.join(fea_data_path, "data.json")):
            for idx, drop_col_id in enumerate(experiment.dataset['drop_columns_ids']):
                TRI_X = experiment.dataset['train_inputs'][:, drop_col_id]
                TE1_X = experiment.dataset['test_inputs'][:, drop_col_id]
                VAL_X = experiment.dataset['val_inputs'][:, drop_col_id]
                TRI_Y = experiment.dataset['train_outputs']
                TE1_Y = experiment.dataset['test_outputs']
                VAL_Y = experiment.dataset['val_outputs']
                combined_series_X = np.concatenate([TRI_X, TE1_X, VAL_X])
                combined_series_Y = np.concatenate([TRI_Y, TE1_Y, VAL_Y])
                unique_num = min(len(np.unique(combined_series_X)), 200)
                import minepy as minepy
                mine = minepy.MINE(alpha=0.6, c=15)
                mine.compute_score(combined_series_X, combined_series_Y)
                ori_mic = mine.mic()
                best_d = -1
                best_mic = ori_mic
                bset_bins = []
                for d in range(2, unique_num):
                    combined_series_binned, bins = pd.qcut(combined_series_X, q=d, labels=False, retbins=True, duplicates='drop')
                    mine.compute_score(combined_series_binned, combined_series_Y)
                    new_mic = mine.mic()
                    if new_mic > best_mic:
                        best_d = d
                        if abs(new_mic - best_mic) < 1e-9:
                            break
                        best_mic = new_mic
                        bset_bins = bins
                drop_col_idx_2_d[str(drop_col_id)] = best_d
                drop_col_idx_2_bin[str(drop_col_id)] = bset_bins
                if best_d != -1:
                    print(f'best_d: {best_d}, best_mic: {best_mic}')
            with open(os.path.join(fea_data_path, "data.json"), "w") as file:
                json.dump(drop_col_idx_2_d, file)

        else:
            with open(os.path.join(fea_data_path, "data.json"), 'r') as file:
                drop_col_idx_2_d = json.load(file)

        for idx, drop_col_id in enumerate(experiment.dataset['drop_columns_ids']):
            if drop_col_idx_2_d.get(str(drop_col_id), -1) == -1:
                continue
            else:
                if os.path.exists(os.path.join('dataset', params.dataset.lower(), params.dataset_col, 'rf_data', params.rf_name)):
                    all_drop_idx_data = np.concatenate([experiment.dataset['train_inputs'][:, drop_col_id], 
                                                        experiment.dataset['test_inputs'][:, drop_col_id], 
                                                        experiment.dataset['val_inputs'][:, drop_col_id],
                                                        experiment.dataset['rf_data_input'][:, drop_col_id]])
                    combined_series_binned, bins = pd.qcut(all_drop_idx_data, q=drop_col_idx_2_d.get(str(drop_col_id), -1), labels=False, retbins=True, duplicates='drop')
                    X = combined_series_binned[:len(experiment.dataset['train_inputs'])]
                    Y = combined_series_binned[len(experiment.dataset['train_inputs']):len(experiment.dataset['train_inputs']) + len(experiment.dataset['test_inputs'])]
                    W = combined_series_binned[len(experiment.dataset['train_inputs']) + len(experiment.dataset['test_inputs']):len(experiment.dataset['train_inputs']) + len(experiment.dataset['test_inputs']) + len(experiment.dataset['val_inputs'])]
                    Z = combined_series_binned[len(experiment.dataset['train_inputs']) + len(experiment.dataset['test_inputs']) + len(experiment.dataset['val_inputs']):]
                    experiment.dataset['train_inputs'][:, drop_col_id] = X
                    experiment.dataset['test_inputs'][:, drop_col_id] = Y
                    experiment.dataset['val_inputs'][:, drop_col_id] = W
                    experiment.dataset['rf_data_input'][:, drop_col_id] = Z
                else:
                    all_drop_idx_data = np.concatenate([experiment.dataset['train_inputs'][:, drop_col_id], 
                                                        experiment.dataset['test_inputs'][:, drop_col_id], 
                                                        experiment.dataset['val_inputs'][:, drop_col_id]])
                    combined_series_binned, bins = pd.qcut(all_drop_idx_data, q=drop_col_idx_2_d.get(str(drop_col_id), -1), labels=False, retbins=True, duplicates='drop')
                    X = combined_series_binned[:len(experiment.dataset['train_inputs'])]
                    Y = combined_series_binned[len(experiment.dataset['train_inputs']):len(experiment.dataset['train_inputs']) + len(experiment.dataset['test_inputs'])]
                    W = combined_series_binned[len(experiment.dataset['train_inputs']) + len(experiment.dataset['test_inputs']):]
                    experiment.dataset['train_inputs'][:, drop_col_id] = X
                    experiment.dataset['test_inputs'][:, drop_col_id] = Y
                    experiment.dataset['val_inputs'][:, drop_col_id] = W
                

        if params.rf:
            rf_data = np.load(
                os.path.join('dataset', params.dataset.lower(), params.dataset_col, 'rf_data', 'rf.npy'),
                allow_pickle=True)
            for item in rf_data:
                ori_input = experiment.dataset['test_inputs'][item.source_id]
                for idx, col in enumerate(experiment.dataset['drop_columns_ids']):
                    item.input[col] = ori_input[col]
            rf_data_input = [item.input for item in rf_data]
            experiment.dataset['new_rf_data_input'] = np.array(rf_data_input)
            experiment.dataset['new_rf_data_output'] = [item.label for item in rf_data]

            
    def cal_gain(choose_col_set, choose_col):
        combined_series_X = combined_data[:, choose_col]
        mine = minepy.MINE(alpha=0.6, c=15)
        mine.compute_score(combined_series_X, combined_series_Y)
        choose_col_2_y_mic = mine.mic()
        choose_col_mic_list = []

        for col in choose_col_set:
            combined_series_AL_C = combined_data[:, col]
            mine.compute_score(combined_series_X, combined_series_AL_C)
            choose_col_mic_list.append(mine.mic())
        for x in v1_inputs.T:
            mine.compute_score(x, combined_series_X)
            mic = mine.mic()
            choose_col_mic_list.append(mic)
        gain = choose_col_2_y_mic - np.mean(choose_col_mic_list)

        return gain
    if params.need_fea_Sel:
        if not os.path.exists(os.path.join(fea_data_path, 'new_col_conflict.npy')): 
            drop_columns = []
            for drop_col in experiment.dataset['drop_columns_ids']:
                drop_columns.append(combined_data[:, drop_col])
            mic_num_table = [0 for _ in range(len(experiment.dataset['drop_columns_ids']))]
            for idx, combined_series_X in enumerate(drop_columns):
                mine = minepy.MINE(alpha=0.6, c=15)
                mine.compute_score(combined_series_X, combined_series_Y)
                mic_num_table[idx] = mine.mic()
                choose_col_mic_list = []
                for x in v1_inputs.T:
                    mine.compute_score(x, combined_series_X)
                    mic = mine.mic()
                    choose_col_mic_list.append(mic)
                mic_num_table[idx] = mic_num_table[idx] - np.mean(choose_col_mic_list)
            mic_rank = np.argsort(mic_num_table)[::-1]
            choose_col_set = set()

            T = 0
            for idx, choose_col in enumerate(mic_rank):
                if idx == 0:
                    choose_col_set.add(experiment.dataset['drop_columns_ids'][choose_col])
                    continue
                gain = cal_gain(choose_col_set, experiment.dataset['drop_columns_ids'][choose_col])
                if gain >= T:
                    T = T + 1e-5
                    choose_col_set.add(experiment.dataset['drop_columns_ids'][choose_col])

            drop_col_list = list(set(experiment.dataset['drop_columns_ids'])- choose_col_set)

            drop_col_index = drop_col_list
            experiment.dataset['new_col_conflict'] = drop_col_index
            np.save(os.path.join(fea_data_path, 'new_col_conflict.npy'), experiment.dataset['new_col_conflict'])
            drop_col_num = len(drop_col_index)


            experiment.dataset['train_inputs'] = np.delete(experiment.dataset['train_inputs'], drop_col_index, axis=1)
            experiment.dataset['test_inputs'] = np.delete(experiment.dataset['test_inputs'], drop_col_index, axis=1)
            experiment.dataset['val_inputs'] = np.delete(experiment.dataset['val_inputs'], drop_col_index, axis=1)

            experiment.dataset['drop_columns_ids'] = list(set(experiment.dataset['drop_columns_ids'])- set(drop_col_index))

            for idy, j in enumerate(experiment.dataset['drop_columns_ids']):
                k = 0
                for i in drop_col_index:
                    if j >= i:
                        k += 1
                experiment.dataset['drop_columns_ids'][idy] = j - k
            
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            sc.fit(experiment.dataset['train_inputs'])
            experiment.dataset['scaler2'] = sc
            joblib.dump(experiment.dataset['scaler2'], '{}/new_sc_v2.pkl'.format(fea_data_path))

        else:   
            experiment.dataset['new_col_conflict'] = np.load(os.path.join(fea_data_path, 'new_col_conflict.npy'))
            if len(experiment.dataset['new_col_conflict']) != 0:
                experiment.dataset['train_inputs'] = np.delete(experiment.dataset['train_inputs'], experiment.dataset['new_col_conflict'], axis=1)
                experiment.dataset['test_inputs'] = np.delete(experiment.dataset['test_inputs'], experiment.dataset['new_col_conflict'], axis=1)
                experiment.dataset['val_inputs'] = np.delete(experiment.dataset['val_inputs'], experiment.dataset['new_col_conflict'], axis=1)

            experiment.dataset['scaler2'] = joblib.load(os.path.join(fea_data_path, 'new_sc_v2.pkl'))
            experiment.dataset['old_scaler2'] = joblib.load(os.path.join(experiment.dataset['v2_dataset_path'], 'sc_v2.pkl'))
            experiment.dataset['drop_columns_ids'] = list(set(experiment.dataset['drop_columns_ids'])- set(experiment.dataset['new_col_conflict']))
            drop_col_num = len(experiment.dataset['new_col_conflict'])
            experiment.dataset['new_col_conflict'] = sorted(experiment.dataset['new_col_conflict'])
            for idy, j in enumerate(experiment.dataset['drop_columns_ids']):
                k = 0
                for i in experiment.dataset['new_col_conflict']:
                    if j >= i:
                        k += 1
                experiment.dataset['drop_columns_ids'][idy] = j - k
                

    else:
        drop_col_num = 0
    
    
    print(f'new_add_num: {len(experiment.dataset["drop_columns_ids"])}')

    print(f'drop_col_num: {drop_col_num}')
    if params.model == "mobile_price":
        inputs = tf.keras.Input(shape=(20 - drop_col_num,))
        x = layers.Dense(32, activation='relu', name='dense1')(inputs)
        x = layers.Dense(16, activation='relu', name='dense2')(x)
        outputs = layers.Dense(4, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    elif params.model == "fetal_health":
        inputs = tf.keras.Input(shape=(21-drop_col_num,))
        x = layers.Dense(256, activation='relu', name='dense3')(inputs)
        x = layers.Dense(128, activation='relu', name='dense4')(x)
        x = layers.Dense(56, activation='relu', name='dense5')(x)
        outputs = layers.Dense(3, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    elif params.model == "foreast":
        inputs = tf.keras.Input(shape=(54-drop_col_num,))
        x = layers.Dense(128, activation='relu', name='dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='dense4')(x)
        x = layers.Dense(32, activation='relu', name='dense5')(x)
        outputs = layers.Dense(7, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    elif params.model == "diabetes":
        inputs = tf.keras.Input(shape=(8-drop_col_num,))
        x = layers.Dense(32, activation='relu', name='dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='dense2')(x)
        x = layers.Dense(32, activation='relu', name='dense3')(x)
        x = layers.Dense(16, activation='relu', name='dense4')(x)
        outputs = layers.Dense(2, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    elif params.model == "customerchurn":
        inputs = tf.keras.Input(shape=(19-drop_col_num,))
        x = layers.Dense(32, activation='relu', name='dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='dense2')(x)
        x = layers.Dense(32, activation='relu', name='dense3')(x)
        x = layers.Dense(16, activation='relu', name='dense4')(x)
        outputs = layers.Dense(2, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    elif params.model == "onlineshoppers":
        inputs = tf.keras.Input(shape=(17 - drop_col_num,))
        x = layers.Dense(32, activation='relu', name='dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='dense2')(x)
        x = layers.Dense(32, activation='relu', name='dense3')(x)
        x = layers.Dense(16, activation='relu', name='dense4')(x)
        outputs = layers.Dense(2, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    elif params.model == "glass":
        inputs = tf.keras.Input(shape=(9- drop_col_num,))
        x = layers.Dense(32, activation='relu', name='dense1')(inputs)
        x = layers.Dense(128, activation='relu', name='dense2')(x)
        x = layers.Dense(256, activation='relu', name='dense3')(x)
        x = layers.Dense(128, activation='relu', name='dense4')(x)
        outputs = layers.Dense(6, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    elif params.model == "musicgenres":
        inputs = tf.keras.Input(shape=(26- drop_col_num,))
        x = layers.Dense(52, activation='relu', name='dense1')(inputs)
        x = layers.Dense(128, activation='relu', name='dense2')(x)
        x = layers.Dense(64, activation='relu', name='dense4')(x)
        x = layers.Dense(32, activation='relu', name='dense5')(x)
        outputs = layers.Dense(10, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    elif params.model == "hand_gesture":
        inputs = tf.keras.Input(shape=(64-drop_col_num,))
        x = layers.Dense(34, activation='relu', name='dense1')(inputs)
        x = layers.Dense(17, activation='relu', name='dense2')(x)
        outputs = layers.Dense(4, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    elif params.model == "bean":
        inputs = tf.keras.Input(shape=(16-drop_col_num,))
        x = layers.Dense(128, activation='relu', name='dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='dense2')(x)
        x = layers.Dense(32, activation='relu', name='dense3')(x)
        outputs = layers.Dense(7, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    elif params.model == "patient":
        inputs = tf.keras.Input(shape=(10-drop_col_num,))
        x = layers.Dense(32, activation='relu', name='dense1')(inputs)
        x = layers.Dense(64, activation='relu', name='dense2')(x)
        x = layers.Dense(32, activation='relu', name='dense3')(x)
        outputs = layers.Dense(2, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    elif params.model == "climate":
        inputs = tf.keras.Input(shape=(13-drop_col_num,))
        x = layers.Dense(26, activation='relu', name='dense1')(inputs)
        x = layers.Dense(32, activation='relu', name='dense2')(x)
        x = layers.Dense(64, activation='relu', name='dense3')(x)
        x = layers.Dense(32, activation='relu', name='dense4')(x)
        x = layers.Dense(17, activation='relu', name='dense5')(x)
        outputs = layers.Dense(9, name='beforesoftmax')(x)
        outputs = layers.Softmax(name='predictions')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    else:
        raise Exception("Unknown Model:" + str(params.model))
    

    return model


def _get_model(params, experiment):
    import os
    model = tf.keras.models.load_model(os.path.join('models', params.dataset.lower(), params.dataset_col, 'v1_weights.hdf5'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

