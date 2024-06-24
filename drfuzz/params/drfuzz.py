import itertools

from params.parameters import Parameters

drfuzz = Parameters()

drfuzz.K = 64
drfuzz.batch1 = 1
drfuzz.batch2 = 1
drfuzz.p_min = 0.0
drfuzz.gamma = 5
drfuzz.alpha = 0.02
drfuzz.beta = 0.20
drfuzz.TRY_NUM = 50
drfuzz.MIN_FAILURE_SCORE = -100
drfuzz.framework_name = 'drfuzz'

drfuzz.save_batch = False