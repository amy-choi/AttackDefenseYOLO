# encoding:utf-8
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ["train", "val"]

classes = ["car", "cyclist", "pedestrian"]


def convert_annotation(image_id, out_file):
    in_file = open("data/Annotations/%s.xml" % (image_id), encoding="utf-8")
    tree = ET.parse(in_file)
    root = tree.getroot()
    objects = root.findall("object")

    for obj in objects:
        cls = obj.find("name").text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (
            int(float(xmlbox.find("xmin").text)),
            int(float(xmlbox.find("ymin").text)),
            int(float(xmlbox.find("xmax").text)),
            int(float(xmlbox.find("ymax").text)),
        )
        out_file.write(" " + ",".join([str(a) for a in b]) + "," + str(cls_id))


wd = getcwd()

for image_set in sets:
    image_ids = open("data/ImageSets/Main/%s.txt" % (image_set)).read().strip().split()
    list_file = open("%s_kitti.txt" % (image_set), "w", encoding="utf-8")  
    for image_id in image_ids:
        list_file.write("%s/data/JPEGImages/%s.jpg" % (wd, image_id))  
        convert_annotation(image_id, list_file)
        list_file.write("\n")
    list_file.close()

# os.system("cat train.txt val.txt > train_val.txt")
# os.system("cat train.txt val.txt test.txt > train_all.txt")
