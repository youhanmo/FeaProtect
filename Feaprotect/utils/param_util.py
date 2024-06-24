import importlib
from src.utility import merge_object

def load_params(params):
    for params_set in params.params_set:
        m = importlib.import_module("params." + params_set)
        new_params = getattr(m, params_set)
        params = merge_object(params, new_params)
    return params


def get_params():
    import argparse
    parser = argparse.ArgumentParser(description="Feature Processing")
    parser.add_argument("--dataset", type=str, default="climate")
    parser.add_argument("--dataset_col", type=str,
                        default="latitude-longitude-region_id-province_id")
    parser.add_argument("--model", type=str, default="climate")
    parser.add_argument("--fix_type", type=str, default='eg_discrt_fea_sel_1')
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--rf", action='store_true')
    parser.add_argument("--rf_name", type=str, default="retrain.npy")
    parser.add_argument("--need_eg", action='store_true')
    parser.add_argument("--need_fea_Sel", action='store_true')
    params = parser.parse_args()

    return params
