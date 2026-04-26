import os
import json
import argparse
import numpy as np
import xml.dom.minidom as XML

voc_cls_id = {"aeroplane":0, "bicycle":1, "bird":2, "boat":3, "bottle":4,
               "bus":5, "car":6, "cat":7, "chair":8, "cow":9,
               "diningtable":10, "dog":11, "horse":12, "motorbike":13, "person":14,
               "pottedplant":15, "sheep":16, "sofa":17, "train":18, "tvmonitor":19}

def get_label(data_path):
    print("generating labels for VOC07 dataset")
    xml_paths = os.path.join(data_path, "VOC2007/Annotations/")
    save_dir = "data/voc07/labels"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in os.listdir(xml_paths):
        if not i.endswith(".xml"):
            continue
        s_name = i.split('.')[0] + ".txt"
        s_dir = os.path.join(save_dir, s_name)
        xml_path = os.path.join(xml_paths, i)
        DomTree = XML.parse(xml_path)
        Root = DomTree.documentElement

        obj_all = Root.getElementsByTagName("object")
        leng = len(obj_all)
        cls = []         # class
        difi_tag = []    # label difficulty
        for obj in obj_all:
            # get the classes
            obj_name = obj.getElementsByTagName('name')[0]
            one_class = obj_name.childNodes[0].data
            cls.append(voc_cls_id[one_class])

            difficult = obj.getElementsByTagName('difficult')[0]
            difi_tag.append(difficult.childNodes[0].data)

        for i, c in enumerate(cls):
            with open(s_dir, "a") as f:
                f.writelines("%s,%s\n" % (c, difi_tag[i]))


def transdifi(data_path):
    print("generating final json file for VOC07 dataset")
    label_dir = "data/voc07/labels/"      # use previous get_label function  get  .txt
    img_dir = os.path.join(data_path, "VOC2007/JPEGImages/")

    # get trainval test id
    id_dirs = os.path.join(data_path, "VOC2007/ImageSets/Main/")
    f_train = open(os.path.join(id_dirs, "train.txt"), "r").readlines()
    f_val = open(os.path.join(id_dirs, "val.txt"), "r").readlines()
    f_trainval = f_train + f_val
    f_test = open(os.path.join(id_dirs, "test.txt"), "r")

    trainval_id =  np.sort([int(line.strip()) for line in f_trainval]).tolist()
    test_id = [int(line.strip()) for line in f_test]
    trainval_data = []
    test_data = []

    # ternary label
    # -1 means negative
    # 0 means difficult
    # +1 means positive

    # binary label
    # 0 means negative #困难样本被设置成0， 可能表示 不参加训练。。
    # +1 means positive #正常样本  ； 困难：他是把所有标注困难的样本 没有来训练吗！

    # we use binary labels in our implementation
    # 根据难度标记决定目标数组target中对应位置的值：
    # 若类别存在且难度非全为1，则标记为1（正样本）；若难度全为1，
    # 则标记为0（困难样本，处理为负样本）。

    for item in sorted(os.listdir(label_dir)):
        with open(os.path.join(label_dir, item), "r") as f:

            target = np.array([-1] * 20)  #（代表20个类别的负样本）
            classes = []  #存储类别ID，
            diffi_tag = [] #存储难度标记。

            for line in f.readlines():
                cls, tag = map(int, line.strip().split(','))
                classes.append(cls)
                diffi_tag.append(tag)

            classes = np.array(classes)
            diffi_tag = np.array(diffi_tag)
            for i in range(20):  #生成类别目标数组：
                if i in classes:
                    i_index = np.where(classes == i)[0] #获取当前类别i在classes中的所有位置索引
                    #根据i_index的长度（即类别i出现的次数）和难度标记（diffi_tag）来更新目标数组target：
                    if len(i_index) == 1:
                        target[i] = 1 - diffi_tag[i_index]
                    else:
                        if len(i_index) == sum(diffi_tag[i_index]):
                            target[i] = 0
                            # 由于不是所有实例都是标注难度的样本
                            # target设置成1， 表示图像里面存在这个 类别的
                        else:
                            target[i] = 1
                else:
                    continue
            img_path = os.path.join(img_dir, item.split('.')[0]+".jpg")
            # 对于这张图像，我们得到的target数组将会是长度为20的数组，其中大部分位置是-1（表示没
            # 有这个类别的信息），而在ID为7, 11, 和 14 的位置分别是0, 1, 和 1，表示该图像的类
            # 别信息和难度标记。
            #eg 假设 cat 标注文件里面难度是1， 这里算完 在target【i】就成为 0了

            #分配数据到训练/验证集或测试集：
            if int(item.split('.')[0]) in trainval_id:
                target[target == -1] = 0  # from ternary to binary by treating difficult as negatives
                data = {"target": target.tolist(), "img_path": img_path}   #target前面是array ，要变成list
                trainval_data.append(data)
            if int(item.split('.')[0]) in test_id:
                data = {"target": target.tolist(), "img_path": img_path}      
                test_data.append(data)

    json.dump(trainval_data, open("data/voc07/trainval_voc07.json", "w"))
    json.dump(test_data, open("data/voc07/test_voc07.json", "w"))
    print("VOC07 data preparing finished!")
    print("data/voc07/trainval_voc07.json data/voc07/test_voc07.json")
    
    # remove label directory   ； 删去之前中转的  标注txt文件
    for item in os.listdir(label_dir):
        os.remove(os.path.join(label_dir, item))
    os.rmdir(label_dir)

# We treat difficult classes in trainval_data as negtive while ignore them in test_data
# The ignoring operation can be automatically done during evaluation (testing).
# The final json file include: trainval_voc07.json & test_voc07.json
# which is the following format:
# [item1, item2, item3, ......,]
# item1 = {
#      "target": 
#      "img_path":      
# }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Usage: --data_path   /your/dataset/path/VOCdevkit
    parser.add_argument("--data_path", default="Dataset/VOCdevkit/", type=str, help="The absolute path of VOCdevkit")
    args = parser.parse_args()

    if not os.path.exists("data/voc07"):
        os.makedirs("data/voc07")
    
    if 'VOCdevkit' not in args.data_path:
        print("WARNING: please include \'VOCdevkit\' str in your args.data_path")
        # exit()

    get_label(args.data_path)
    transdifi(args.data_path)

