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
    parser = argparse.ArgumentParser(description="Experiments Script For DeepReFuzz")
    parser.add_argument("--params_set", nargs='*', type=str,
                        default=["Dnn", "mobilePrice", "change", "drfuzz"],
                        help="see params folder")

    parser.add_argument("--dataset", type=str, default="mobile_price")

    parser.add_argument("--dataset_col", type=str,
                        default="Insulin-BMI-Age")

    parser.add_argument("--model", type=str, default="mobile_price")
    parser.add_argument("--coverage", type=str, default="change", choices=["change", "neuron", "kmn", "nbc", "snac"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--update_col_num", type=int, default=3)
    parser.add_argument('--choose_col_type_list', nargs='+', help='Array argument', default="")
    parser.add_argument("--choose_col_type", type=str, default='expected_gradients_source', choices=['random',
                                                                                'expected_gradients_source',
                                                                                'MIC'
                                                                                ])
    parser.add_argument("--time_interval", type=int, default=30, help="time_interval")
    parser.add_argument("--time", type=int, default=2)
    parser.add_argument("--terminate_type", type=str, default="time", choices=["time", "iteration"])
    parser.add_argument("--model2_type", type=str, default="ori")
    parser.add_argument("--output_name", type=str, default='price_range', help="output name")
    parser.add_argument("--max_iteration", type=int, default=2000)
    parser.add_argument("--iteration_interval", type=int, default=20)
    params = parser.parse_args()
    params = load_params(params)
    params.time_minutes = params.time
    params.time_period = params.time_minutes * 60
    f_threshold_dict = {'mobile_price': 1.5, 'fetal_health': 2,
                        'customerchurn': 0.4, 'diabetes': 2,
                        'bean': 1.5, 'hand_gesture': 0.72,
                        'patient': 0.5, 'musicgenres': 0.62,
                        "climate": 0.4}
    params.f_threshold = f_threshold_dict[params.dataset]
    return params
