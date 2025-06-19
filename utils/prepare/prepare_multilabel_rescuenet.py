import os
import json
import argparse
import numpy as np
import xml.dom.minidom as XML
from PIL import Image

# cls_id0 = {
#  1: 'Water',
#  2: 'Building_No_Damage',
#  3: 'Building_Minor_Damage',
#  4: 'Building_Major_Damage',
#  5: 'Building_Total_Destruction',
#  6: 'Vehicle',
#  7: 'Road-Clear',
#  8: 'Road-Blocked',
#  9: 'Tree',
#  10: 'Pool'
# }     # besides, background:0


def get_label(data_path):     #Dataset/rescuenet/
    print("generating labels for rescuenet dataset")
    anno_paths1 = os.path.join(data_path, "segmentation-trainset/train-label-img/")
    anno_paths2 = os.path.join(data_path, "segmentation-validationset/val-label-img/")
    anno_paths3 = os.path.join(data_path, "segmentation-testset/test-label-img/")
    save_dir = "data/rescuenet/labels"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path=[anno_paths1,anno_paths2,anno_paths3]
    for subpath in path:
        for image in os.listdir(subpath):
            imagepath = os.path.join(subpath, image)
            s_name = image.split('.')[0] + ".txt"
            s_dir = os.path.join(save_dir, s_name)

            annotation_image = Image.open(imagepath)
            annotation_array = np.array(annotation_image)

            unique_classes = np.unique(annotation_array)

            leng = len(unique_classes)
            cls = []  # class
            for one_class in unique_classes:
                cls.append(one_class)
            for i, c in enumerate(cls):
                with open(s_dir, "a") as f:
                    f.writelines("%s\n" % cls[i])

def transdifi(data_path):  #Dataset/rescuenet/
    print("generating final json file for rescuenet dataset")
    label_dir = "data/rescuenet/labels/"
    img_dir = os.path.join(data_path, "allimage/")

    # get trainval test id
    f_train = open(os.path.join(data_path, "train.txt"), "r").readlines()
    f_val = open(os.path.join(data_path, "val.txt"), "r").readlines()
    # f_trainval = f_train + f_val
    f_test = open(os.path.join(data_path, "test.txt"), "r")

    train_id = [int(line.strip()) for line in f_train]
    val_id = [int(line.strip()) for line in f_val]
    test_id = [int(line.strip()) for line in f_test]

    # trainval_id = np.sort([int(line.strip()) for line in f_trainval]).tolist()
    train_data, val_data, test_data = [], [], []

    for item in sorted(os.listdir(label_dir)):
        with open(os.path.join(label_dir, item), "r") as f:
            target = np.array([0] * 10)
            classes = []
            for line in f.readlines():
                cls = line.strip()
                classes.append(int(cls))

            # classes = np.array(classes)
            for k in range(10):
                i=k+1
                if i in classes:
                    target[k] = 1
                else:
                    continue

            image_name0=item.split('.')[0]
            image_name=image_name0[:-4]
            img_path = os.path.join(img_dir, image_name + ".jpg")

            if int(image_name) in train_id:
                data = {"target": target.tolist(), "img_path": img_path}
                train_data.append(data)
            if int(image_name) in val_id:
                data = {"target": target.tolist(), "img_path": img_path} 
                val_data.append(data)
            if int(image_name) in test_id:
                data = {"target": target.tolist(), "img_path": img_path}
                test_data.append(data)

    json.dump(train_data, open("data/rescuenet/train_rescuenet.json", "w"))
    json.dump(val_data, open("data/rescuenet/val_rescuenet.json", "w"))
    json.dump(test_data, open("data/rescuenet/test_rescuenet.json", "w"))
    print("rescuenet data preparing finished!")
    print("data/rescuenet/train_rescuenet.json  data/rescuenet/val_rescuenet.json data/rescuenet/test_rescuenet.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Usage: --data_path   /your/dataset/path/VOCdevkit
    parser.add_argument("--data_path", default="Dataset/rescuenet/", type=str, help="The absolute path of VOCdevkit")
    args = parser.parse_args()

    if not os.path.exists("data/rescuenet"):
        os.makedirs("data/rescuenet")
    # get_label(args.data_path)
    transdifi(args.data_path)

