import numpy as np

from utils.logger import logger
logger = logger(__name__)



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
