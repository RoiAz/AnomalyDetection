import pandas as pd
import numpy as np
from os import path


class resultAccuracy:
    def __init__(self, labels_path, skip=None, num_of_rows=None, threshold=10):
        if not path.exists(labels_path):
            raise Exception("path - " + labels_path + " doesn't exists")
        labels_df = pd.read_csv(labels_path, skiprows=skip, nrows=num_of_rows, dtype=np.int8, header=None,
                                  usecols=[1])
        self.labels = labels_df.to_numpy(dtype=bool, copy=True).flatten()
       # print("#"*10)
        #print(labels_df)
        #print(self.labels)
        self.threshold = threshold
        self.num_of_success = 0
        self.num_of_packets = 0
        self.true_positive = 0  # match - detect the attack
        self.false_positive = 0  # false alarm - thought the packet is malicious but it isn't
        self.false_negative = 0  # didn't detect the attack
        self.true_negative = 0  # match - thought the packet isn't malicious and that's the case
        self.success_rate = 0

    def add(self, rmse, index):
        is_real_malicious = self.labels[index-1]
        is_predicted_malicious = False
        self.num_of_packets += 1
        if rmse >= self.threshold:
            is_predicted_malicious = True
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
        return self.success_rate

    def accuracyRate(self):
        self.success_rate = self.num_of_success / self.num_of_packets
        return self.success_rate
