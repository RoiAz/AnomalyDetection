import pandas as pd
import numpy as np
from os import path

class logger:
    def __init__(self, filename_path):
        self.loglist = []
        self.textfile_path = filename_path

    def add(self, message):
        self.loglist.append(message)

    def print_to_file(self):
        textfile = open(self.textfile_path, "w")
        for element in a_list:
            textfile.write(element + "\n")
        textfile.close()
