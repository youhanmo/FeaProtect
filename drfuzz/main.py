import torch
torch.device("cpu")
from src.DrFuzz import DrFuzz
from src.experiment_builder import get_experiment
from utils.param_util import get_params
import sys
sys.path.append('autoencoder')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from utils.logger import logger
logger = logger(__name__)

import warnings
warnings.filterwarnings("ignore")
import os,sys
os.chdir(sys.path[0])
if __name__ == '__main__':
    params = get_params()
    experiment = get_experiment(params)

    logger.info(f'PARAMS.choose_col_type: {params.choose_col_type}')
    experiment.time_list = [i * params.time_interval for i in range(1, params.time // params.time_interval + 1 + 1)]
    import numpy as np
    import os
    experiment_dir = str(params.coverage)
    dir_name = 'experiment_' + str(params.framework_name)
    res_save_path = os.path.join(dir_name, experiment_dir, params.dataset, params.dataset_col, params.model2_type,
                                 str(params.fidelity_model), str(params.choose_col_type),
                                 str(params.f_threshold), str(params.update_col_num))
    if not os.path.exists(res_save_path):
        os.makedirs(res_save_path)

    params.res_save_path = res_save_path

    dh = DrFuzz(params, experiment)


    both_fail, regression_faults, weaken = dh.run()

    logger.info(f'TOTAL BOTH: {len(both_fail)}')
    logger.info(f'TOTAL REGRESSION: {len(regression_faults)}')
    logger.info(f'TOTAL WEAKEN: {len(weaken)}')
    logger.info(f'CORPUS {dh.corpus}')
    logger.info(f'ITERATION {dh.experiment.iteration}')

    logger.info(f'SCORE {dh.experiment.coverage.get_failure_type()}')

    import pandas as pd

    np.save(os.path.join(res_save_path, str(params.time) + "_bf.npy"), np.asarray(both_fail))
    np.save(os.path.join(res_save_path, str(params.time) + "_rf.npy"), np.asarray(regression_faults))
    result = pd.DataFrame({'AT': params.time,
                            'both_fail': len(both_fail), 'regression_faults': len(regression_faults),
                            'weaken': len(weaken),
                            'corpus': dh.corpus, 'iteration': dh.experiment.iteration,
                            'score': dh.experiment.coverage.get_current_score(),
                            'ftype': dh.experiment.coverage.get_failure_type()}, index=[0])
    data = pd.read_csv(os.path.join(res_save_path, "result.csv"))
    if (data['AT'] == params.time).any():
        pass
    else:
        result.to_csv(os.path.join(res_save_path, "result.csv"), mode='a', index=False, header=False)