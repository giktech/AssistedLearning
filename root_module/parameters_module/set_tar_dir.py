import os
from root_module.parameters_module import Directory


class TargetNetworkDirectory(Directory):
    def __init__(self, mode):
        Directory.__init__(self, mode)

        self.data_path += '/tar'

        ''' DATASET '''
        self.raw_data_filename = self.data_path + self.raw_data_filename
        self.data_filename = self.data_path + self.data_filename
        self.gold_label_filename = self.data_path + self.gold_label_filename   # For valid gold label is used as weak label
        self.weak_label_filename = self.data_path + self.weak_label_filename   # For train, only weak label

    def makedir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
