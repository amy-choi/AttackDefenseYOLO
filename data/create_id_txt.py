# create_txt.py
# encoding:utf-8
from __future__ import print_function
import pdb
import glob
import os
import random
import math


def get_sample_value(txt_name, category_name):
    label_path = "./label_2/"
    txt_path = label_path + txt_name + ".txt"
    try:
        with open(txt_path) as r_tdf:
            if category_name in r_tdf.read():
                return " 1"
            else:
                return "-1"
    except IOError as ioerr:
        print("File error:" + str(ioerr))


txt_list_path = glob.glob("./label_2/*.txt")
txt_list = []

for item in txt_list_path:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    txt_list.append(temp1)
txt_list.sort()
print(txt_list, "\n\n")

# There is a blog suggesting train:val:test=8:1:1, try it first
num_train = txt_list[0 : math.floor(len(txt_list) * 0.5)]  # can change the percentage
print(len(num_train), "\n\n")

num_val = list(set(txt_list).difference(set(num_train)))
print(len(num_val), "\n\n")

# pdb.set_trace()

Main_path = "./ImageSets/Main/"
train_test_name = ["train", "val"]
category_name = ["car", "cyclist", "pedestrian"]

# Cyclically write trainvl train val test
for item_train_test_name in train_test_name:
    list_name = "num_"
    list_name += item_train_test_name
    train_test_txt_name = Main_path + item_train_test_name + ".txt"
    try:
        # Write a single file
        with open(train_test_txt_name, "w") as w_tdf:
            # Write line by line
            for item in eval(list_name):
                w_tdf.write(item + "\n")
        # Cyclically write person car truck
        for item_category_name in category_name:
            category_txt_name = (
                Main_path + item_category_name + "_" + item_train_test_name + ".txt"
            )
            with open(category_txt_name, "w") as w_tdf:
                # Write line by line
                for item in eval(list_name):
                    w_tdf.write(
                        item + " " + get_sample_value(item, item_category_name) + "\n"
                    )
    except IOError as ioerr:
        print("File error:" + str(ioerr))