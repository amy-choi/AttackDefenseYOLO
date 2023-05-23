#!/usr/bin/env python
# encoding: utf-8

import glob
import string

"""
 After take txt files into xml files, create folders in the same directory Annotations
"""

txt_list = glob.glob("./label_2/*.txt")  # Labels store all txt file folder path


def show_category(txt_list):
    category_list = []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(" ")
                    category_list.append(
                        labeldata[0]
                    )  # as long as the first field, i.e. category
        except IOError as ioerr:
            print("File error:" + str(ioerr))
    print(set(category_list))  # output set


def merge(line):
    each_line = ""
    for i in range(len(line)):
        if i != (len(line) - 1):
            each_line = each_line + line[i] + " "
        else:
            each_line = each_line + line[i]  # last field is no space behind
    each_line = each_line + "\n"
    return each_line


print("before modify categories are:\n")
show_category(txt_list)

for item in txt_list:
    new_txt = []
    try:
        with open(item, "r") as r_tdf:
            for each_line in r_tdf:
                labeldata = each_line.strip().split(" ")
                if labeldata[0] in [
                    "Van",
                    "Tram",
                    "Car",
                    "Truck",
                ]:  # combined Automotive
                    labeldata[0] = labeldata[0].replace(labeldata[0], "car")
                if labeldata[0] == "Pedestrian":
                    labeldata[0] = labeldata[0].replace(labeldata[0], "pedestrian")
                if labeldata[0] == "Cyclist":
                    labeldata[0] = labeldata[0].replace(labeldata[0], "cyclist")
                if labeldata[0] in ["Person_sitting", "DontCare"]:
                    continue
                if labeldata[0] == "Misc":  # Class Misc Ignore
                    continue
                new_txt.append(merge(labeldata))  # to re-write the new txt file
        with open(
            item, "w+"
        ) as w_tdf:  # w + is to open the original file contents will be deleted, and the other to write new content into it
            for temp in new_txt:
                w_tdf.write(temp)
    except IOError as ioerr:
        print("File error:" + str(ioerr))

print("\nafter modify categories are:\n")
show_category(txt_list)