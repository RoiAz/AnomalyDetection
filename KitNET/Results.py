import pandas as pd
import numpy as np
from os import path

from KitNET.logger import logger


class resultAccuracy:
    def __init__(self, labels_path, skip=None, num_of_rows=None, threshold=10):
        self.mal_cnt = 0
        if not path.exists(labels_path):
            raise Exception("path - " + labels_path + " doesn't exists")
        labels_df = pd.read_csv(labels_path, skiprows=skip, nrows=num_of_rows, header=None, usecols=[1])
        self.labels = labels_df.to_numpy(dtype=bool, copy=True).flatten()
        # print("#"*10)
        # print(labels_df)
        # print(self.labels)
        self.threshold = threshold
        self.num_of_success = 0
        self.num_of_packets = 0
        self.true_positive = 0  # match - detect the attack
        self.false_positive = 0  # false alarm - thought the packet is malicious but it isn't
        self.false_negative = 0  # didn't detect the attack
        self.true_negative = 0  # match - thought the packet isn't malicious and that's the case
        self.success_rate = 0
        self.malicious_alert = 0
        self.malicious_count = 0
        self.verbose = 0
#        self.logger = logger(r'C:\Users\roeihers\PycharmProjects\AnomalyDetection\accuracy.txt', big_data_mode=1)


    def add(self, rmse, index):
        size = self.labels.size
        if index >= size:
            print("Index too big cant add to resultAccuracy, index: " + str(index)+ " size: "+ str(size))
            return 0
        is_real_malicious = self.labels[index-1]
        if is_real_malicious:
            self.mal_cnt += 1
        is_predicted_malicious = False
        self.num_of_packets += 1
        if rmse >= self.threshold:
            is_predicted_malicious = True
            self.malicious_count +=1
            if self.malicious_count >= 20:
                self.malicious_alert = 1

        if is_predicted_malicious and is_real_malicious:
            success = True
            self.true_positive += 1
        elif is_predicted_malicious and not is_real_malicious:
            success = False
            self.false_positive += 1
        elif not is_predicted_malicious and is_real_malicious:
            success = False
            self.false_negative += 1
        else:
            success = True
            self.true_negative += 1

        if success:
            self.num_of_success += 1
        self.success_rate = self.num_of_success / self.num_of_packets
   #     self.logger.add_rate(self.success_rate, 'success_rate')
        return self.success_rate

    def accuracyRate(self):
        self.success_rate = self.num_of_success / self.num_of_packets
        return self.success_rate

    def truePositiveAccuracyRate(self):
        return self.true_positive / self.mal_cnt


    def maliciousAlert(self):
        if self.malicious_alert == 1 and self.verbose == 1:
            print('Malicious Alert')
         #   self.logger.add_rate(self.success_rate, 'Malicious Alert')
            self.malicious_alert = 0

    def print_rate_to_file(self):
        pass
        #self.logger.print_to_file()