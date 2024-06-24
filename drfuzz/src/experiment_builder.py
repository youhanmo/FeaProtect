import os
import random
import time
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import change_measure_utils as ChangeMeasureUtils
import joblib
from utils.logger import logger
logger = logger(__name__)

class Experiment:
    pass


def get_experiment(params):
    experiment = Experiment()
    experiment.dataset = _get_dataset(params, experiment)
    experiment.model = _get_model(params, experiment)
    experiment.modelv2 = _get_model_v2(params, experiment)
    experiment.coverage = _get_coverage(params, experiment)
    experiment.start_time = time.time()
    experiment.iteration = 0
    experiment.termination_condition = generate_termination_condition(experiment, params)
    experiment.time_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                            210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360]
    return experiment


def generate_termination_condition(experiment, params):
    start_time = experiment.start_time
    time_period = params.time_period
    def termination_condition():
        c2 = time.time() - start_time > time_period
        return c2

    return termination_condition


def _get_dataset(params, experiment):
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
    v1_columns = [col for col in train_df.columns if col not in drop_columns and col != params.output_name]
    v2_columns = [col for col in train_df.columns if col != params.output_name]
    drop_columns_ids = [v2_columns.index(col) for col in drop_columns]
    random_fix_col_list = [col for col in range(len(v2_columns)) if col not in drop_columns_ids]
    random_fix_col = random.sample(random_fix_col_list, params.update_col_num)
    dataset_name = params.dataset.lower()
    logger.info(f'load dataset {params.dataset} successfully, train_inputs shape: {test_inputs.shape}')
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
        'random_fix_col': random_fix_col,
        "dataset_name": dataset_name
    }


def _get_model_v2(params, experiment):
    import os
    model = keras.models.load_model(os.path.join('models', params.dataset.lower(), params.dataset_col, params.model2_type, 'v2_weights.hdf5'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def _get_model(params, experiment):
    import os
    model = keras.models.load_model(os.path.join('models', params.dataset.lower(), params.dataset_col, 'v1_weights.hdf5'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def _get_coverage(params, experiment):
    def input_scaler(test_inputs):
        model_lower_bound = params.model_input_scale[0]
        model_upper_bound = params.model_input_scale[1]
        input_lower_bound = params.input_lower_limit
        input_upper_bound = params.input_upper_limit
        scaled_input = (test_inputs - input_lower_bound) / (input_upper_bound - input_lower_bound)
        scaled_input = scaled_input * (model_upper_bound - model_lower_bound) + model_lower_bound
        return scaled_input

    if params.coverage == "change":
        from coverages.change_scorer import ChangeScorer
        # TODO: Skip layers should be determined autoamtically

        coverage = ChangeScorer(params, experiment.model, experiment.modelv2, threshold=0.5,
                                skip_layers=ChangeMeasureUtils.get_skiped_layer(experiment.model))  # 0:input, 5:flatten

    else:
        raise Exception("Unknown Coverage" + str(params.coverage))


    return coverage
