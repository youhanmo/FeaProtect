import collections
import datetime
import gc
import time
import traceback
import random
import torch
torch.device("cpu")
import numpy as np
import pandas as pd
import tensorflow as tf
import src.testcase_utils as TestCaseUtils
from src.DrFuzz_struct_mutop import get_mutation_func
from src.mutation_selection_logic import MCMC, Roulette
import utils.struct_util as StructUtil
import utils.fidelity_util as FidelityUtil
from sklearn.metrics import mutual_info_score
from scipy.stats import kendalltau, weightedtau
import utils.expect_grad_ops_util as eager_ops
from sklearn.feature_selection import mutual_info_regression
now_mutator_names = []
import os
import pickle
from utils.logger import logger
from minepy import MINE
logger = logger(__name__)

class INFO:
    def __init__(self):
        self.dict = {}

    def __getitem__(self, i):
        _i = str(i)
        if _i in self.dict:
            return self.dict[_i]
        else:
            I0, I0_new, state = np.copy(i), np.copy(i), 0
            return I0, I0_new, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]


class TestCase:
    def __init__(self, input, ground_truth, source_id):
        self.input = input
        self.label = ground_truth
        self.source_id = source_id
        self.generation = 0
        self.exploration_multiple = 1
        self.ori_input = input
        self.ref_input = input
        self.m1_trace = []
        self.m2_trace = []
        self.m1_predict_label = -1
        self.m2_predict_label = -1
        self.save_in_queue_time = -1
        self.mutation_trace = []
        self.corpus_id = -1
        self.last_corpus_trace = []
        self.fidelity = -1
        self.col_contribution = []
        self.col_select = []
        self.select_prob = self.calculate_select_prob(self.generation, 0)

    def update_ref(self, new_ref_input):
        self.ref_input = new_ref_input

    def update_save_in_queue_time(self, official_saved_time):
        self.save_in_queue_time = official_saved_time


    def set_trace(self, new_m1_trace, new_m2_trace):
        self.m1_trace = new_m1_trace
        self.m2_trace = new_m2_trace
        self.m1_predict_label = np.argmax(self.m1_trace)
        self.m2_predict_label = np.argmax(self.m2_trace)

    def get_test_failure_tend(self):
        if self.m1_predict_label == self.label and self.m1_predict_label == self.m2_predict_label and self.m1_predict_label != -1 and self.m2_predict_label != -1:
            failure_trace = self.m2_trace - self.m1_trace
            failure_trace[self.m1_predict_label] = -1
            failure_direction = np.argmax(failure_trace)
            return (self.source_id, self.m1_predict_label, failure_direction)
        else:
            if self.m1_predict_label == self.label and self.m2_predict_label != self.label:
                return 'rf'
            return None

    def get_test_tend(self):
        return (self.source_id, self.m1_predict_label, self.m2_predict_label)

    def get_all_test_failure_tends_dicts(self):
        classnum = len(self.m1_trace)
        failure_trace = self.m2_trace - self.m1_trace
        failure_trace_dicts = {}
        for i in range(classnum):
            if self.m1_predict_label == self.label and self.m2_predict_label != i:
                failure_trace_dicts[(self.source_id, self.m1_predict_label, i)] = failure_trace[i]
        return failure_trace_dicts

    def get_difference(self):
        return (self.m1_trace[self.label] - self.m2_trace[self.label])

    def get_relative_difference(self):
        epsilon = 1e-7
        return (self.m1_trace[self.label] - self.m2_trace[self.label]) / (self.m1_trace[self.label] + epsilon)

    def calculate_select_prob(self, generation, select_times, init=1, m=20, finish=0.05):
        delt_times = 10
        if select_times > delt_times:
            return 0

        alpha = np.log(init / finish) / m
        l = - np.log(init) / alpha
        decay = np.exp(-alpha * (float(generation + select_times / 2.0) + l))

        if decay == np.nan:
            decay = 0
        return decay

def testcaselist2nparray(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].input))
    new_input = np.asarray(new_input)
    return new_input

def testcaselist2sourceid(test_case_list, gran='category'):
    new_input = []
    if gran == 'category':
        for i in range(len(test_case_list)):
            new_input.append(np.asarray(test_case_list[i].source_id))
    new_input = np.asarray(new_input)
    return new_input


def testcaselist2labels(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].label))
    new_input = np.asarray(new_input)
    return new_input

def testcaselist2generation(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].generation))
    new_input = np.asarray(new_input)
    return new_input

def timing(start_time, duriation):
    MIN = 60
    duriation_sec = duriation * MIN
    now = time.time()
    return now - start_time > duriation_sec


class DrFuzz:
    def __init__(self, params, experiment):
        self.params = params
        self.experiment = experiment
        self.info = INFO()
        self.last_coverage_state = None
        self.m2_input_shape = params.m2_input_shape
        self.pass_prob = 1
        self.corpus = 0
        self.corpus_list = []
        self.both_failure_case = []
        self.regression_faults = []
        self.weaken_case = []
        self.last_used_mutator = None
        self.fidelity_model, self.fidelity_model_name = FidelityUtil.load_fidelity_model(self.params, experiment)
        self.f_threshold = params.f_threshold
        self.mutation_strategy_mode = self.params.mutation_strategy_mode
        self.feature_contribution_table = collections.defaultdict(list)
        self.init_contribution(params.choose_col_type)
        self.rf_source_id_failure_diversity = collections.defaultdict(list)
        if self.mutation_strategy_mode == 'MCMC':
            self.mutation_strategy = MCMC()
        else:
            self.mutation_strategy = Roulette()

    def scale_to_unit_interval(self, lst):
        max_value = max(lst)
        min_value = min(lst)
        scaled_lst = [(x - min_value) / (max_value - min_value) for x in lst]
        return scaled_lst

    def to_probability_table(self, lst, epsilon=1e-6):
        scaled_lst = self.scale_to_unit_interval(lst)
        shifted_lst = [x + epsilon for x in scaled_lst]
        total = sum(shifted_lst)
        return [x / total for x in shifted_lst]

    def init_contribution(self, type):
        I_input = np.concatenate(
            (self.experiment.dataset['train_inputs'], self.experiment.dataset['test_inputs']), axis=0)
        I_label = np.concatenate(
            (self.experiment.dataset['train_outputs'], self.experiment.dataset['test_outputs']), axis=0)
        preprocessed_input = StructUtil.preprocess_data(I_input, self.experiment.dataset['scaler2'])
        m1_preprocessed_input = np.delete(preprocessed_input, self.experiment.dataset['drop_columns_ids'],
                                          axis=1)
        m2_preprocessed_input = preprocessed_input
        m1_prob_vector = self.experiment.model.predict(m1_preprocessed_input)
        self.experiment.dataset['num_class'] = len(m1_prob_vector[0])
        m2_prob_vector = self.experiment.modelv2.predict(m2_preprocessed_input)
        m1_result = np.argmax(m1_prob_vector, axis=1)
        m2_result = np.argmax(m2_prob_vector, axis=1)
        I_label = np.squeeze(I_label)
        m1_result = np.squeeze(m1_result)
        m2_result = np.squeeze(m2_result)
        regression_fault = []
        fix_fault = []
        both_truth = []
        both_truth_label = []
        for index in range(len(I_label)):
            if I_label[index] == m1_result[index] and I_label[index] != m2_result[index]:
                regression_fault.append(m2_preprocessed_input[index])
            elif I_label[index] != m1_result[index] and I_label[index] == m2_result[index]:
                fix_fault.append(m2_preprocessed_input[index])
            elif I_label[index] == m1_result[index] and I_label[index] == m2_result[index]:
                both_truth.append(m2_preprocessed_input[index])
                both_truth_label.append(I_label[index])
        self.m1_both_truth = np.delete(both_truth, self.experiment.dataset['drop_columns_ids'],
                                       axis=1)
        self.m1_fix_fault = np.delete(fix_fault, self.experiment.dataset['drop_columns_ids'], axis=1)
        self.m1_regression_fault = np.delete(regression_fault,
                                             self.experiment.dataset['drop_columns_ids'], axis=1)
        self.m2_both_truth = np.array(both_truth)
        self.m2_fix_fault = np.array(fix_fault)
        self.m2_regression_fault = np.array(regression_fault)
        _, _, is_enum, _, _ = StructUtil.get_low_high(self.experiment.dataset['train_df'], self.experiment.dataset['dataset_name'])
        idx2col = {idx: str(item) for idx, item in enumerate(self.experiment.dataset['v2_columns'])}

        if type in ['random']:
            self.m1_feature_importance_explainer = None
            self.m2_feature_importance_explainer = None
        
        elif type in ['MIC']:
            self.m1_feature_importance_explainer = None
            self.m2_feature_importance_explainer = None
            mine = MINE(alpha=0.6, c=15)
            train_inputs = StructUtil.preprocess_data(self.experiment.dataset['train_inputs'],
                                                      self.experiment.dataset['scaler2'])
            drop_columns = []
            for drop_col in self.experiment.dataset['drop_columns_ids']:
                drop_columns.append(train_inputs[:, drop_col])
            train_inputs = np.delete(train_inputs, self.experiment.dataset['drop_columns_ids'], axis=1)
            conflict_num_table = [0 for _ in range(len(self.experiment.dataset['v1_columns']))]
            for y in drop_columns:
                for idx, x in enumerate(train_inputs.T):
                    mine.compute_score(x, y)
                    mic = mine.mic()
                    conflict_num_table[idx] += mic

            prob_table = self.to_probability_table(conflict_num_table)
            prob_table = self.fill_data(np.expand_dims(prob_table, axis=0),
                                        (1, len(self.experiment.dataset['v2_columns'])),
                                        self.experiment.dataset['drop_columns_ids'],
                                        type='zero')[0]
            self.prob_table = prob_table

        elif type in ['expected_gradients_source']:
            self.m1_feature_importance_explainer = None
            self.m2_feature_importance_explainer = None
        
        
        else:
            raise Exception('type error')
    

    def fill_data(self, data, data2_shap, fill_idx, type='mean'):
        new_data = np.zeros(data2_shap)
        mask = np.ones(data2_shap[1], dtype=bool)
        mask[fill_idx] = False
        new_data[:, mask] = data
        if type == 'mean':
            new_data[:, ~mask] = np.mean(data, axis=1)[:, np.newaxis]
        elif type == 'zero':
            new_data[:, ~mask] = 0
        return new_data

    def compute_contribution(self, tc):
        if self.params.choose_col_type in ['random']:
            prob_table = [random.uniform(0, 1) for _ in range(len(self.experiment.dataset['v1_columns']))]
            prob_table = self.fill_data(np.expand_dims(prob_table, axis=0),
                                        (1, len(self.experiment.dataset['v2_columns'])),
                                        self.experiment.dataset['drop_columns_ids'],
                                        type='zero')[0]
            total_prob = sum(prob_table)
            prob_table = [p / total_prob for p in prob_table]
            tc.col_contribution = prob_table

        elif self.params.choose_col_type in ['MIC']:
            tc.col_contribution = self.prob_table
        
        elif self.params.choose_col_type in ['expected_gradients_source']:
            m2_background_summary = self.m2_both_truth
            m1_background_summary = np.delete(m2_background_summary, self.experiment.dataset['drop_columns_ids'], axis=1)
            m1_background_summary = np.array(m1_background_summary, dtype=np.float32)
            m2_background_summary = np.array(m2_background_summary, dtype=np.float32)
            truth_label = tc.label
            truth_label = tf.constant(truth_label, dtype=tf.int32)
            preprocessed_input = StructUtil.preprocess_data(np.expand_dims(tc.input, axis=0),
                                                            self.experiment.dataset['scaler2'])
            m1_preprocessed_input = tf.constant(np.delete(preprocessed_input, self.experiment.dataset['drop_columns_ids'],
                                              axis=1), dtype=tf.float32)
            m2_preprocessed_input = tf.constant(preprocessed_input, dtype=tf.float32)

            m1_value = eager_ops.expected_gradients_full(m1_preprocessed_input, 
                                                         m1_background_summary, 
                                                         self.experiment.model,
                                                         k=100, index_true_class=True, 
                                                         labels=np.expand_dims(
                                                         tf.keras.utils.to_categorical(truth_label, num_classes=self.experiment.dataset['num_class']), 
                                                        axis=0))
            m2_value = eager_ops.expected_gradients_full(m2_preprocessed_input, 
                                                         m2_background_summary, 
                                                         self.experiment.modelv2,
                                                         k=100, index_true_class=True, 
                                                         labels=np.expand_dims(
                                                         tf.keras.utils.to_categorical(truth_label, num_classes=self.experiment.dataset['num_class']), 
                                                         axis=0))
            new_m2_val = np.delete(m2_value, self.experiment.dataset['drop_columns_ids'], axis=1)
            prob_table = new_m2_val - m1_value
            prob_table = self.to_probability_table(prob_table[0])
            prob_table = self.fill_data(np.expand_dims(prob_table, axis=0),
                                            (1, len(self.experiment.dataset['v2_columns'])),
                                            self.experiment.dataset['drop_columns_ids'],
                                            type='zero')[0]
            tc.col_contribution = prob_table

    def termination_condition(self, start_time):
        if self.params.terminate_type == 'time':
            c2 = time.time() - start_time > self.params.time_period
        elif self.params.terminate_type == 'iteration':
            c2 = self.experiment.iteration > self.params.max_iteration
        return c2

    def run(self):
        starttime = time.time()

        I_input = self.experiment.dataset["test_inputs"]
        I_label = self.experiment.dataset["test_outputs"]
        scaler2 = self.experiment.dataset['scaler2']
        preprocessed_input = StructUtil.preprocess_data(I_input, scaler2)
        fidelity_list = FidelityUtil.compute_fidelity(self.fidelity_model, self.fidelity_model_name, preprocessed_input)
        fidelity_list = np.squeeze(fidelity_list)
        m1_preprocessed_input = np.delete(preprocessed_input, self.experiment.dataset['drop_columns_ids'], axis=1)
        m2_preprocessed_input = preprocessed_input
        m1_prob_vector = self.experiment.model.predict(m1_preprocessed_input, batch_size=16)
        m2_prob_vector = self.experiment.modelv2.predict(m2_preprocessed_input, batch_size=16)
        m1_result = np.argmax(m1_prob_vector, axis=1)
        I_label = np.squeeze(I_label)
        m1_result = np.squeeze(m1_result)
        good_idx = np.where(I_label == m1_result)
        good_idx = good_idx[0]
        I_list = []
        for i, (input, label, fidelity, m1_prob, m2_prob) in enumerate(
                zip(I_input, I_label, fidelity_list, m1_prob_vector, m2_prob_vector)):
            if i in good_idx:
                tc = TestCase(input, label, i)
                self.rf_source_id_failure_diversity[(i, label)] = [1 for _ in range(len(m1_prob))]
                tc.update_save_in_queue_time(time.time() - starttime)
                tc.set_trace(new_m1_trace=m1_prob, new_m2_trace=m2_prob)
                self.compute_contribution(tc)
                tc.fidelity = fidelity
                tc.corpus_id = self.corpus
                I_list.append(tc)
                self.corpus_list.append(tc)
                self.corpus += 1

        _, regression_faults_in_initial_seeds, rest_list = self.experiment.coverage.initial_seed_list(I_list)

        self.regression_faults.extend(regression_faults_in_initial_seeds)
        for rfis in regression_faults_in_initial_seeds:
            rfis.update_save_in_queue_time(time.time() - starttime)

        T = self.Preprocess(rest_list)
        B, B_id = self.SelectNext(T)

        time_list = self.experiment.time_list
        self.experiment.iteration = 0
        total_time = self.params.time * 60 
        while not self.termination_condition(starttime):
            if self.experiment.iteration % 100 == 0:
                gc.collect()
            self.update_prob(self.experiment.iteration)
            S = B
            B_new = []
            count_regression = []
            count_both_fail = []
            count_weaken = []
            count_halfweaken = []
            count_fix = []
            if len(time_list) > 0:
                if timing(starttime, time_list[0]):
                    result = pd.DataFrame({'AT': str(time_list[0]),
                                            'both_fail': len(self.both_failure_case),
                                            'regression_faults': len(self.regression_faults),
                                            'weaken': len(self.weaken_case),
                                            'corpus': self.corpus, 'iteration': self.experiment.iteration,
                                            'score': self.experiment.coverage.get_current_score(),
                                            'ftype': self.experiment.coverage.get_failure_type()}, index=[0])
                    if not os.path.exists(os.path.join(self.params.res_save_path, "result.csv")):
                        result.to_csv(os.path.join(self.params.res_save_path, "result.csv"), index=False)
                    else:
                        result.to_csv(os.path.join(self.params.res_save_path, "result.csv"), mode='a',
                                        index=False, header=False)

                    np.save(os.path.join(self.params.res_save_path, str(time_list[0]) + '_rf'),
                            self.regression_faults)
                    np.save(os.path.join(self.params.res_save_path, str(time_list[0]) + '_bf'),
                            self.both_failure_case)
                    time_list.remove(time_list[0])
                if len(time_list) == 0:
                    import sys
                    sys.exit(0)

            Mutants = []
            try:
                for s_i in range(len(S)):
                    I = S[s_i]
                    if self.fidelity_model_name in ['tave', 'vae', 'ctgan', 'auto_encoder']:
                        Mutants.extend(self.mutate_for_discrimitator(I))
                    else:
                        for i in range(1, 20 + 1):
                            I_new = self.Mutate_new(I)
                            if I_new != None and self.isChanged(I, I_new):
                                Mutants.append(I_new)
                if len(Mutants) > 0:
                    bflist, rflist, wklist, hwklist, B_new, dangerous_source_id, fixlist = self.isFailedTestList(I, Mutants)
                    self.both_failure_case.extend(bflist)
                    count_both_fail.extend(bflist)
                    self.regression_faults.extend(rflist)
                    count_regression.extend(rflist)
                    self.weaken_case.extend(wklist)
                    count_weaken.extend(wklist)
                    count_halfweaken.extend(hwklist)
                    count_fix.extend(fixlist)

                if self.mutation_strategy_mode == 'MCMC':
                    k = 1 / 3
                    for mt_ in B_new + count_regression:
                        mt_dif = mt_.get_difference()
                        m2_dif = I.m2_trace[I.label] - mt_.m2_trace[mt_.label]
                        delta = (1 - k) * m2_dif + k * mt_dif
                        if delta > 0.0:
                            self.mutation_strategy.mutators[mt_.mutation_trace[-1]].difference_score_total += 1
                            break

                selected_test_case_list = []
                if len(B_new) > 0:
                    for i in range(len(B_new)):
                        B_new_failure_dicts = B_new[i].get_all_test_failure_tends_dicts()
                        for failure_id in B_new_failure_dicts:
                            if B_new_failure_dicts[failure_id] > self.experiment.coverage.get_failure_score(failure_id):
                                selected_test_case_list.append(B_new[i])
                                break

                if len(selected_test_case_list) > 0 or len(count_regression) > 0:
                    selected_test_case_list_add_rf = []
                    selected_test_case_list_add_rf.extend(selected_test_case_list)
                    selected_test_case_list_add_rf.extend(count_regression)
                    regression_faults_failure_ids, past_mutop, history_corpus_ids_on_rf_branch = TestCaseUtils.testcaselist2pastandfailureid(
                        count_regression)
                    self.experiment.coverage.step(selected_test_case_list_add_rf, update_state=True)

                    for ca in selected_test_case_list:
                        ca.update_save_in_queue_time(time.time() - starttime)

                    B_c, Bs = T
                    for i in range(len(selected_test_case_list)):
                        B_c += [0]
                        selected_test_case_list_i = np.asarray(selected_test_case_list)[i:i + 1]
                        Bs += [selected_test_case_list_i]
                        selected_test_case_list[i].corpus_id = self.corpus
                        self.corpus_list.append(selected_test_case_list[i])
                        self.corpus += 1

                    regression_faults_failure_ids_set = set(regression_faults_failure_ids)
                    history_corpus_ids_on_rf_branch_set = set(history_corpus_ids_on_rf_branch)
                    if len(regression_faults_failure_ids_set) > 0:
                        Bs_tc_id = 0
                        for Bs_tc in Bs:
                            Bs_tc_failure = Bs_tc[0].get_test_failure_tend()
                            if (Bs_tc_failure == 'rf' or Bs_tc_failure in regression_faults_failure_ids_set) and Bs_tc[
                                0].corpus_id in history_corpus_ids_on_rf_branch_set:
                                Bs_tc[0].select_prob = 0
                            if self.experiment.coverage.source_id_banned_dict[Bs_tc[0].source_id] >= 9:
                                Bs_tc[0].select_prob = 0
                            Bs_tc_id += 1

                        self.BatchPrioritize(T, B_id)

                    del selected_test_case_list
                    del count_regression
                    del count_both_fail
                    del count_weaken
                    del count_halfweaken

                self.experiment.iteration += 1
                if self.experiment.iteration % 5000 == 0:
                    if 'random' not in self.params.choose_col_type: 
                        self.update_seed_feature_prop(T)

                B, B_id = self.SelectNext(T)
            except:
                logger.error(traceback.format_exc())
                return self.both_failure_case, self.regression_faults, self.weaken_case

        return self.both_failure_case, self.regression_faults, self.weaken_case

    def update_seed_feature_prop(self, T):
        B_c, Bs = T
        for tc in Bs:
            self.compute_contribution(tc[0])


    def Preprocess(self, I):
        _I = np.random.permutation(I)
        Bs = np.array_split(_I, range(self.params.batch1, len(_I), self.params.batch1))

        return list(np.zeros(len(Bs))), Bs
    
    def SelectNext(self, T):
        B_c, Bs = T
        B_p = [i[0].select_prob for i in Bs]
        epsilon = 1e-7
        c = np.random.choice(len(Bs), p=B_p / (np.sum(B_p) + epsilon))
        return Bs[c], c

    def isFailedTestList(self, I, I_new_list):
        model_v1 = self.experiment.model
        I_new_list_inputs = testcaselist2nparray(I_new_list)
        ground_truth_list = testcaselist2labels(I_new_list)
        I_new_input = I_new_list_inputs.reshape(-1, self.params.m2_input_shape[1])
        I_new_input_preprocess = StructUtil.preprocess_data(I_new_input, self.experiment.dataset['scaler2'])
        I_new_input_preprocess_v1 = np.delete(I_new_input_preprocess, self.experiment.dataset['drop_columns_ids'], axis=1)
        I_new_input_preprocess_v2 = I_new_input_preprocess
        temp_result_v1 = model_v1.predict(I_new_input_preprocess_v1)
        predict_result_v1 = np.argmax(temp_result_v1, axis=1)
        y_prob_vector_max_confidence_m1 = np.max(temp_result_v1, axis=1)
        model_v2 = self.experiment.modelv2
        temp_result_v2 = model_v2.predict(I_new_input_preprocess_v2)
        predict_result_v2 = np.argmax(temp_result_v2, axis=1)
        y_m2_at_m1_max_pos = []
        for i in range(len(temp_result_v2)):
            y_m2_at_m1_max_pos.append(temp_result_v2[i][predict_result_v1[i]])
        difference = (y_prob_vector_max_confidence_m1 - y_m2_at_m1_max_pos)
        difference_I = np.max(I.m1_trace) - I.m2_trace[I.m1_predict_label]

        both_file_list = []
        regression_faults_list = []
        weaken_faults_list = []
        half_weaken_faults_list = []
        rest_case_list = []
        fix_case_list = []
        potential_source_id = []

        for i in range(len(I_new_list)):
            I_new_list[i].set_trace(new_m1_trace=temp_result_v1[i], new_m2_trace=temp_result_v2[i])
            if 'source' in self.params.choose_col_type:
                I_new_list[i].col_contribution = I.col_contribution
            else:
                self.compute_contribution(I_new_list[i])
            if predict_result_v1[i] != ground_truth_list[i] and predict_result_v2[i] != ground_truth_list[i]:
                both_file_list.append(I_new_list[i])
            elif predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] != ground_truth_list[i]:
                self.rf_source_id_failure_diversity[(I_new_list[i].source_id, I_new_list[i].label)][predict_result_v2[i]] += 1
                I_new_list[i].exploration_multiple += 1
                potential_source_id.append(I_new_list[i].source_id)
                regression_faults_list.append(I_new_list[i])
            elif predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[i] and \
                    difference[i] > 0.3:
                I_new_list[i].exploration_multiple += 1
                rest_case_list.append(I_new_list[i])
                weaken_faults_list.append(I_new_list[i])
                potential_source_id.append(I_new_list[i].source_id)
            elif predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[i] and \
                    difference[i] > 0.15:
                I_new_list[i].exploration_multiple += 1
                half_weaken_faults_list.append(I_new_list[i])
                rest_case_list.append(I_new_list[i])
                potential_source_id.append(I_new_list[i].source_id)
            elif predict_result_v1[i] != ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[i]:
                fix_case_list.append(I_new_list[i])
            elif (predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[i] and
                  ((difference_I > 0 > difference[i] and difference_I - difference[i] > 0.15) or (
                          difference[i] < difference_I < 0 and difference_I - difference[i] > 0.15))):
                fix_case_list.append(I_new_list[i])
            else:
                rest_case_list.append(I_new_list[i])
        return both_file_list, regression_faults_list, weaken_faults_list, half_weaken_faults_list, rest_case_list, potential_source_id, fix_case_list

    def isChanged(self, I, I_new):
        return np.any(I.input != I_new.input)
    
    def isNewInputChanged(self, I, I_new):
        return np.any(I.input != I_new)

    def BatchPrioritize(self, T, B_id):
        B_c, Bs = T
        B_c[B_id] += 1
        Bs[B_id][0].select_prob = Bs[B_id][0].calculate_select_prob(Bs[B_id][0].generation, B_c[B_id])

    def mutate_for_discrimitator(self, I):
        I_new_list = []
        I_new_last_mutator_list = []
        now_mutator_name = self.mutation_strategy.choose_mutator(self.last_used_mutator)
        now_mutator = get_mutation_func(now_mutator_name)
        self.last_used_mutator = now_mutator_name
        now_mutator_names.append(now_mutator_name)
        for i in range(1, self.params.TRY_NUM + 1):
            I_new = now_mutator(I, self.experiment.dataset, self.params).reshape(*(self.m2_input_shape[1:]))
            I_new = StructUtil.clip_input(I_new, self.experiment)
            I_new_last_mutator_list.append(now_mutator_name)
            I_new_list.append(I_new)

        I_pre_list = StructUtil.preprocess_data(I_new_list, self.experiment.dataset['scaler2'])
        I_new_fidelity = FidelityUtil.compute_fidelity(self.fidelity_model, self.fidelity_model_name,
                                                       I_pre_list)
        if len(I_new_list) > 1:
            I_new_fidelity = np.squeeze(I_new_fidelity)
        new_case_list = []
        is_mutant_fidelity = False

        for i in range(len(I_new_list)):
            if self.fidelity_check(I_new_fidelity[i], I.fidelity, mode=self.fidelity_model_name) and self.isNewInputChanged(I, I_new_list[i]):
                new_case = TestCase(I_new_list[i], I.label, I.source_id)
                new_case.generation = I.generation + 1
                new_case.mutation_trace.extend(I.mutation_trace)
                new_case.mutation_trace.append(I_new_last_mutator_list[i])
                new_case.select_prob = new_case.calculate_select_prob(new_case.generation, 0)
                new_case.fidelity = I_new_fidelity[i]
                new_case.ori_input = I.ori_input
                new_case.last_corpus_trace.extend(I.last_corpus_trace)
                new_case.last_corpus_trace.append(I.corpus_id)
                new_case_list.append(new_case)
                is_mutant_fidelity = True

        if self.mutation_strategy_mode == 'MCMC':
            self.mutation_strategy.mutators[I_new_last_mutator_list[i]].total_select_times += 1
            if is_mutant_fidelity:
                self.mutation_strategy.mutators[I_new_last_mutator_list[i]].fidelity_case_num += 1

        return new_case_list

    def Mutate_new(self, I):
        for i in range(1, self.params.TRY_NUM + 1):
            now_mutator_name = self.mutation_strategy.choose_mutator(self.last_used_mutator)

            now_mutator = get_mutation_func(now_mutator_name)
            self.last_used_mutator = now_mutator_name
            I_new = now_mutator(np.copy(I.input), self.experiment.dataset).reshape(*(self.m2_input_shape[1:]))
            I_new = StructUtil.clip_input(I_new, self.experiment)

            I_pre_list = StructUtil.preprocess_data(I_new, self.experiment.dataset['scaler2'])
            I_new_fidelity = FidelityUtil.compute_fidelity(self.fidelity_model, self.fidelity_model_name,
                                                           I_pre_list)
            I_new_fidelity = np.squeeze(I_new_fidelity)

            if self.fidelity_check(I_new_fidelity, I.fidelity, mode='ssim') and self.isNewInputChanged(I, I_new):
                new_case = TestCase(I_new, I.label, I.source_id)
                new_case.generation = I.generation + 1
                new_case.mutation_trace.extend(I.mutation_trace)
                new_case.mutation_trace.append(now_mutator_name)
                new_case.fidelity = I_new_fidelity
                new_case.ori_input = I.ori_input
                new_case.last_corpus_trace.extend(I.last_corpus_trace)
                new_case.last_corpus_trace.append(I.corpus_id)
                return new_case


        return None

    def fidelity_check(self, I_new_fidelity, I_fidelity, mode='vae'):
        if mode == 'vae' or mode == 'auto_encoder' or mode == 'tave':
            return I_new_fidelity <= self.f_threshold
        elif mode == 'ctgan':
            return I_new_fidelity >= self.f_threshold
        else:
            return I_new_fidelity >= I_fidelity
        
        
    def update_prob(self, iteration):
        if iteration > 5000:
            self.pass_prob = 1
        elif iteration > 10000:
            self.pass_prob = 1
        elif iteration > 20000:
            self.pass_prob = 1
