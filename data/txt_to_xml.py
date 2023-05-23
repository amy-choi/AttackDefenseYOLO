# txt_to_xml.py
# encoding:utf-8

from xml.dom.minidom import Document
import cv2
import os


def generate_xml(name, split_lines, img_size, class_ind, aug):
    doc = Document()  # Create DOM document object

    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)

    title = doc.createElement("folder")
    title_text = doc.createTextNode("KITTI")
    title.appendChild(title_text)
    annotation.appendChild(title)

    img_name = name + ".jpg"

    title = doc.createElement("filename")
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement("source")
    annotation.appendChild(source)

    title = doc.createElement("database")
    title_text = doc.createTextNode("The KITTI Database")
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement("annotation")
    title_text = doc.createTextNode("KITTI")
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement("size")
    annotation.appendChild(size)

    title = doc.createElement("width")
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement("height")
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement("depth")
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for split_line in split_lines:
        line = split_line.strip().split()
        if line[0] in class_ind:
            object = doc.createElement("object")
            annotation.appendChild(object)

            title = doc.createElement("name")
            title_text = doc.createTextNode(line[0])
            title.appendChild(title_text)
            object.appendChild(title)

            bndbox = doc.createElement("bndbox")
            object.appendChild(bndbox)
            title = doc.createElement("xmin")
            title_text = doc.createTextNode(str(int(float(line[4]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement("ymin")
            title_text = doc.createTextNode(str(int(float(line[5]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement("xmax")
            title_text = doc.createTextNode(str(int(float(line[6]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement("ymax")
            title_text = doc.createTextNode(str(int(float(line[7]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)

    # Write the DOM object doc to the file
    if aug:
        f = open("./Annotations/" + name + ".xml", "a+")  # create a new xml file
        f.write(doc.toprettyxml(indent=""))
    else:
        f = open("./Annotations/" + name + ".xml", "w")  # create a new xml file
        f.write(doc.toprettyxml(indent=""))
    f.close()


# #source code
if __name__ == "__main__":
    class_ind = ("car", "cyclist", "pedestrian")
    # cur_dir=os.getcwd()  # current path
    # labels_dir=os.path.join(cur_dir,'labels') # get the current path and build a new path.and the result is'../yolo_learn/labels'
    labels_dir = "./label_2"
    for parent, dirnames, filenames in os.walk(
        labels_dir
    ):  # Get the root directory, subdirectory and files in the root directory respectively
        for file_name in filenames:
            full_path = os.path.join(parent, file_name)  # Get the full path of the file
            f = open(full_path)
            split_lines = f.readlines()
            name = file_name[
                :-4
            ]  # is after four extensions .txt, taking only the first file name
            aug = False

            img_name = name + ".jpg"
            img_path = os.path.join(
                "./JPEGImages", img_name
            )  # The path needs to be modified by yourself
            print(img_path)
            img_size = cv2.imread(img_path).shape
            generate_xml(name, split_lines, img_size, class_ind, aug)

    for parent, dirnames, filenames in os.walk(
        labels_dir
    ):  # Get the root directory, subdirectory and files in the root directory respectively
        for file_name in filenames:
            full_path = os.path.join(parent, file_name)  # Get the full path of the file
            f = open(full_path)
            split_lines = f.readlines()
            name = file_name[
                :-4
            ]  # is after four extensions .txt, taking only the first file name
            name += "_aug"
            aug = True

            img_name = name + ".jpg"
            img_path = os.path.join(
                "./JPEGImages", img_name
            )  # The path needs to be modified by yourself
            print(img_path)
            img_size = cv2.imread(img_path).shape
            generate_xml(name, split_lines, img_size, class_ind, aug)

print("all txts has converted into xmls")