import os
import numpy as np
import torch
device = torch.device("cpu")


def load_fidelity_model(params, experiment):
    model_params = params.fidelity_model.split('-')
    model_name, model_configuration = model_params[0], model_params[1]
    model_path = os.path.join('autoencoder/model', params.dataset, model_name, model_configuration, 'model.h5').replace('\\', '/')
    if torch.cuda.is_available():
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location=torch.device('cpu'))
    model.decoder.eval()
    model.encoder.eval()
    return model, model_name


def compute_fidelity(model, model_name, I_input):
    pred_list = model.get_reconstruction(I_input)
    pred_list = np.array(pred_list)
    mae = np.mean(np.abs(pred_list - I_input), axis=1)
    mse = np.mean(np.square(pred_list - I_input), axis=1)
    return mae
