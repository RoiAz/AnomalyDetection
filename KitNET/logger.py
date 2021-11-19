import pandas as pd
import numpy as np
from os import path

class logger:
    def __init__(self, filename_path,big_data_mode):
        self.big_dataset_counter = 0
        self.mess_num = 0
        self.loglist = []
        self.textfile_path = filename_path
        self.big_dataset_mode = big_data_mode #if using big dataset put on 1
      #  self.textfile = open(self.textfile_path, "w")


    def add(self, message):
        if (self.big_dataset_mode == 0) or ( (self.big_dataset_counter % 1000) == 0):
            self.loglist.append(message)
            self.mess_num+=1
        if (self.big_dataset_mode == 1):
                self.big_dataset_counter += 1

    def add_rate(self, value , rate):
        if (self.big_dataset_mode == 0) or ( (self.big_dataset_counter % 1000) == 0):
            message = '[' + str(self.mess_num) + '][RATE]' + ' current ' + rate + ' is: ' + str(value)
            self.loglist.append(message)
            self.mess_num+=1
        if(self.big_dataset_mode == 1):
            self.big_dataset_counter+=1

    def add_packet(self, i , rmse):
        if (self.big_dataset_mode == 0) or ( (self.big_dataset_counter % 1000) == 0):
            message = '[' + str(self.mess_num) + '][PACKET]' + 'packet index: --- ' + str(i) + " --- have rmse of " + str(rmse)
            self.loglist.append(message)
            self.mess_num+=1
        if(self.big_dataset_mode == 1):
            self.big_dataset_counter+=1
    def print_to_file(self):
        print('printing')
        for element in self.loglist:
            self.textfile.write(element + "\n")
        self.textfile.close()
