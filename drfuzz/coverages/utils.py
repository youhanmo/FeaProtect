import traceback
from os import path, makedirs
import h5py
import numpy as np
import sys
# from cleverhans.attacks import SaliencyMapMethod, FastGradientMethod, CarliniWagnerL2, BasicIterativeMethod
# from cleverhans.utils_keras import KerasModelWrapper
import tensorflow as tf
from tensorflow.keras import backend as K

from math import ceil
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow.keras as keras

def get_layer_outs(model, test_input, skip=[]):
    inp = model.input  # input placeholder
    outputs = [layer.output for index, layer in enumerate(model.layers) \
               if index not in skip]

    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layer_outs = [func([test_input]) for func in functors]

    return layer_outs


def get_layer_outs_new(params, model, test_input, skip=[]):
    # print([layer.output for index, layer in enumerate(model.layers)
    #                                   if index not in skip][0:])
    if params.model == "LeNet5_prune" or params.model == "vgg16_prune" or params.model == "resnet18_prune" or params.model == "Alexnet_prune":
        import tensorflow
        evaluator = tensorflow.keras.models.Model(inputs=model.input,
                                                  outputs=[layer.output for index, layer in enumerate(model.layers)
                                                           if index not in skip][0:])  # original is 1
    else:
        evaluator = keras.models.Model(inputs=model.input,
                                       outputs=[layer.output for index, layer in enumerate(model.layers)
                                                if index not in skip][0:])
    return evaluator.predict(test_input)


def load_major_func_regions(picklefile):
    major_regions = []
    import pickle
    output = open(picklefile, 'rb+')
    major_regions = pickle.load(output)
    output.close()

    return major_regions


def calc_major_func_regions(params, model, train_inputs, skip=None):
    if skip is None:
        skip = []
    outs = get_layer_outs_new(params, model, train_inputs, skip)

    major_regions = []

    for layer_index, layer_out in enumerate(outs):
        layer_out = layer_out.mean(axis=tuple(i for i in range(1, layer_out.ndim - 1)))

        major_regions.append((layer_out.min(axis=0), layer_out.max(axis=0)))

    return major_regions


def get_layer_outputs_by_layer_name(model, test_input, skip=None):
    if skip is None:
        skip = []

    inp = model.input  # input placeholder
    outputs = {layer.name: layer.output for index, layer in enumerate(model.layers)
               if (index not in skip and 'input' not in layer.name)}
    functors = {name: K.function([inp], [out]) for name, out in outputs.items()}

    layer_outs = {name: func([test_input]) for name, func in functors.items()}
    return layer_outs


def get_layer_inputs(model, test_input, skip=None, outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs(model, test_input)

    inputs = []

    for i in range(len(outs)):
        weights, biases = model.layers[i].get_weights()

        inputs_for_layer = []

        for input_index in range(len(test_input)):
            inputs_for_layer.append(
                np.add(np.dot(outs[i - 1][0][input_index] if i > 0 else test_input[input_index], weights), biases))

        inputs.append(inputs_for_layer)

    return [inputs[i] for i in range(len(inputs)) if i not in skip]


def get_python_version():
    if (sys.version_info > (3, 0)):
        return 3
    else:
        return 2



def load_perturbed_test(filename):
    # read X
    with h5py.File(filename + '_perturbations_x.h5', 'r') as hf:
        x_perturbed = hf["x_perturbed"][:]

    # read Y
    with h5py.File(filename + '_perturbations_y.h5', 'r') as hf:
        y_perturbed = hf["y_perturbed"][:]

    return x_perturbed, y_perturbed


def save_perturbed_test_groups(x_perturbed, y_perturbed, filename, group_index):
    # save X
    filename = filename + '_perturbations.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        group.create_dataset("x_perturbed", data=x_perturbed)
        group.create_dataset("y_perturbed", data=y_perturbed)

    print("Classifications saved in ", filename)

    return


def load_perturbed_test_groups(filename, group_index):
    with h5py.File(filename + '_perturbations.h5', 'r') as hf:
        group = hf.get('group' + str(group_index))
        x_perturbed = group.get('x_perturbed').value
        y_perturbed = group.get('y_perturbed').value

        return x_perturbed, y_perturbed


def create_experiment_dir(experiment_path, model_name,
                          selected_class, step_size,
                          approach, susp_num, repeat):
    experiment_name = model_name + '_C' + str(selected_class) + '_SS' + \
                      str(step_size) + '_' + approach + '_SN' + str(susp_num) + '_R' + str(repeat)

    if not path.exists(experiment_path):
        makedirs(experiment_path)

    return experiment_name


def save_classifications(correct_classifications, misclassifications, filename, group_index):
    filename = filename + '_classifications.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        group.create_dataset("correct_classifications", data=correct_classifications)
        group.create_dataset("misclassifications", data=misclassifications)

    print("Classifications saved in ", filename)
    return


def load_classifications(filename, group_index):
    filename = filename + '_classifications.h5'
    print
    filename
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            correct_classifications = group.get('correct_classifications').value
            misclassifications = group.get('misclassifications').value

            print("Classifications loaded from ", filename)
            return correct_classifications, misclassifications
    except (IOError) as error:
        print("Could not open file: ", filename)
        sys.exit(-1)


def save_layer_outs(layer_outs, filename, group_index):
    filename = filename + '_layer_outs.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        for i in range(len(layer_outs)):
            group.create_dataset("layer_outs_" + str(i), data=layer_outs[i])

    print("Layer outs saved in ", filename)
    return


def load_layer_outs(filename, group_index):
    filename = filename + '_layer_outs.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            layer_outs = []
            while True:
                layer_outs.append(group.get('layer_outs_' + str(i)).value)
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except (AttributeError) as error:
        print("Layer outs loaded from ", filename)
        return layer_outs


def save_original_inputs(original_inputs, filename, group_index):
    filename = filename + '_originals.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        group.create_dataset("x_original", data=original_inputs)

    print("Originals saved in ", filename)

    return


def filter_correct_classifications(model, X, Y):
    X_corr = []
    Y_corr = []
    X_misc = []
    Y_misc = []
    for x, y in zip(X, Y):
        p = model.predict(np.expand_dims(x, axis=0))
        if np.argmax(p) == np.argmax(y):
            X_corr.append(x)
            Y_corr.append(y)
        else:
            X_misc.append(x)
            Y_misc.append(y)
    return np.array(X_corr), np.array(Y_corr), np.array(X_misc), np.array(Y_misc)


def filter_val_set(desired_class, X, Y):
    """
    Filter the given sets and return only those that match the desired_class value
    :param desired_class:
    :param X:
    :param Y:
    :return:
    """
    X_class = []
    Y_class = []
    for x, y in zip(X, Y):
        if y[desired_class] == 1:
            X_class.append(x)
            Y_class.append(y)
    print("Validation set filtered for desired class: " + str(desired_class))
    return np.array(X_class), np.array(Y_class)


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def get_trainable_layers(model):
    trainable_layers = []
    for idx, layer in enumerate(model.layers):
        try:
            if 'input' not in layer.name and 'softmax' not in layer.name and \
                    'pred' not in layer.name:
                weights = layer.get_weights()[0]
                trainable_layers.append(model.layers.index(layer))
        except:
            pass

    return trainable_layers


def construct_spectrum_matrices(model, trainable_layers,
                                correct_classifications, misclassifications,
                                layer_outs):
    scores = []
    num_cf = []
    num_uf = []
    num_cs = []
    num_us = []
    for tl in trainable_layers:
        num_cf.append(np.zeros(model.layers[tl].output_shape[1]))
        num_uf.append(np.zeros(model.layers[tl].output_shape[1]))
        num_cs.append(np.zeros(model.layers[tl].output_shape[1]))
        num_us.append(np.zeros(model.layers[tl].output_shape[1]))
        scores.append(np.zeros(model.layers[tl].output_shape[1]))

    for tl in trainable_layers:
        layer_idx = trainable_layers.index(tl)
        all_neuron_idx = range(model.layers[tl].output_shape[1])
        test_idx = 0
        for l in layer_outs[tl][0]:
            covered_idx = list(np.where(l > 0)[0])
            uncovered_idx = list(set(all_neuron_idx) - set(covered_idx))
            if test_idx in correct_classifications:
                for cov_idx in covered_idx:
                    num_cs[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_us[layer_idx][uncov_idx] += 1
            elif test_idx in misclassifications:
                for cov_idx in covered_idx:
                    num_cf[layer_idx][cov_idx] += 1
                for uncov_idx in uncovered_idx:
                    num_uf[layer_idx][uncov_idx] += 1
            test_idx += 1

    return scores, num_cf, num_uf, num_cs, num_us


def cone_of_influence_analysis(model, dominants):
    hidden_layers = [l for l in dominants.keys() if len(dominants[l]) > 0]
    target_layer = max(hidden_layers)

    scores = []
    for i in range(1, target_layer + 1):
        scores.append(np.zeros(model.layers[i].output_shape[1]))

    for i in range(2, target_layer + 1)[::-1]:
        for j in range(model.layers[i].output_shape[1]):
            for k in range(model.layers[i - 1].output_shape[1]):
                relevant_weights = model.layers[i].get_weights()[0][k]
                if (j in dominants[i] or scores[i - 1][j] > 0) and relevant_weights[j] > 0:
                    scores[i - 2][k] += 1
                elif (j in dominants[i] or scores[i - 1][j] > 0) and relevant_weights[j] < 0:
                    scores[i - 2][k] -= 1
                elif j not in dominants[i] and scores[i - 1][j] < 0 and relevant_weights[j] > 0:
                    scores[i - 2][k] -= 1
                elif j not in dominants[i] and scores[i - 1][j] < 0 and relevant_weights[j] < 0:
                    scores[i - 2][k] += 1
    print(scores)
    return scores


def percent(part, whole):
    if part == 0:
        return 0
    return float(part) / whole * 100


def percent_str(part, whole):
    return "{0}%".format(float(part) / whole * 100)
