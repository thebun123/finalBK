import numpy as np
import random
import os
import glob
import shutil
import argparse

DATA_PATH = './data'
DATA_TO_TRAIN_PATH = './data_to_train'
TEST_PATH = './test'
TRAIN_PATH = './train'
VAL_PATH = './val'

parser = argparse.ArgumentParser('Make new data to train model')
parser.add_argument('--make_new', default=False, type=bool, help='make new data')


def compute_label(path):
    list_data_each_label = []
    for i in range(10):
        files = glob.glob('{}/{}_*.wav'.format(path, i))
        list_data_each_label.append(files)
    return list_data_each_label


# train 70 test 30
def make_train_and_test(list_data, flag):
    random.shuffle(list_data)
    flag = flag+ random.randrange(-4, 4)
    # copy to train
    for file in list_data[:flag]:
        dst = DATA_TO_TRAIN_PATH + '/' + file.split('/')[-1]
        shutil.copy(file, dst)

    # copy to test
    for file in list_data[flag:]:
        dst = TEST_PATH + '/' + file.split('/')[-1]
        shutil.copy(file, dst)


def make_train_and_validation(list_data):
    flag = int(0.8 * len(list_data))
    random.shuffle(list_data)

    # copy to train
    for file in list_data[:flag]:
        dst = TRAIN_PATH + '/' + file.split('/')[-1]
        shutil.copy(file, dst)

    # copy to test
    for file in list_data[flag:]:
        dst = VAL_PATH + '/' + file.split('/')[-1]
        shutil.copy(file, dst)


if __name__ == '__main__':
    args = parser.parse_args()
    make_new = args.make_new

    if make_new:
        shutil.rmtree(DATA_TO_TRAIN_PATH)
        shutil.rmtree(TRAIN_PATH)
        shutil.rmtree(VAL_PATH)
        shutil.rmtree(TEST_PATH)
        os.mkdir(DATA_TO_TRAIN_PATH)
        os.mkdir(TRAIN_PATH)
        os.mkdir(TEST_PATH)
        os.mkdir(VAL_PATH)

        # split 70:30 to train and test
        list_data_each_label = compute_label(DATA_PATH)

        total = 0
        for files in list_data_each_label:
            total+= len(files)
        mean = total//10
        for files in list_data_each_label:
            make_train_and_test(files, int(mean*0.7))

        # split 80:20 to train and validation
        list_data_to_train = compute_label(DATA_TO_TRAIN_PATH)
        # total = 0
        # for idx, list_data in enumerate(list_data_to_train):
        #     print(str(idx) + ' : ' + str(len(list_data)))
        #     total+=len(list_data)
        # print(total/100)

        for files in list_data_to_train:
            make_train_and_validation(files)

