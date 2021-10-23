import panda as pd

class resultAccuracy:
    def __init__(self, labels_csv_doc_path):
        self.rmse_index_list = []
        self.label_doc_path = labels_csv_doc_path
        self.limit = 10
        self.success_rate = 0
        self.false_negative= 0
        self.true_negative = 0


    def add(self, rmse, index):
        if rmse >= self.limit:
            self.rmse_index_list.append(index)

    def accuracyrate(self):
        keep_rows = self.rmse_index_list
        labels_file = pd.read_csv(self.label_doc_path, header= 0, skiprows=lambda x: x not in keep_rows)

        for list_member in rmse_index_list:
            tmp_index = 0
            if labels_file[tmp_index] == 1:
                self.true_negative = self.true_negative + 1
            else:
                self.false_negative = self.false_negative + 1
            tmp_index = tmp_index + 1

        self.success_rate =  self.true_negative/(self.true_negative + self.false_negative)

        return self.success_rate