import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from .cal_mAP import json_map
from .cal_PR import json_metric, metric, json_metric_top3,per_class_metric

import os
import datetime
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from math import ceil, sqrt

rescuenet_classes = ('Water', 'Building_No_Damage', 'Building_Minor_Damage',
 'Building_Major_Damage', 'Building_Total_Destruction', 'Vehicle',
              'Road-Clear', 'Road-Blocked', 'Tree', 'Pool')

rescuenet_classes_g1 = ('Building_No_Damage', 'Building_With_Damage','Road-Clear', 'Road-Blocked')

rescuenet_classes_g2 = ('Building_No_Damage', 'Building_With_Damage',
 'Building_Collapsed', 'Road-Clear', 'Road-Blocked')

rescuenet_classes_g3 = ('Building_No_Damage', 'Building_Minor_Damage',
 'Building_Major_Damage', 'Building_Total_Destruction', 'Road-Clear', 'Road-Blocked')

class_dict = {
    "rescuenet": rescuenet_classes,
    "rescuenet_g1": rescuenet_classes_g1,
    "rescuenet_g2": rescuenet_classes_g2,
    "rescuenet_g3": rescuenet_classes_g3
}

# labels = ['Water', 'Building_No_Damage', 'Building_Minor_Damage',
#  'Building_Major_Damage', 'Building_Total_Destruction', 'Vehicle',
#               'Road-Clear', 'Road-Blocked', 'Tree', 'Pool']

def evaluation(matrix,epoch_now,result, types, ann_path,modelname,log_folder_path):
    print("Evaluation")
    classes = class_dict[types]
    num_classes = len(classes)
    aps = np.zeros(len(classes), dtype=np.float64)
    precision_per_class = np.zeros(num_classes, dtype=np.float64)
    recall_per_class = np.zeros(num_classes, dtype=np.float64)
    f1_per_class = np.zeros(num_classes, dtype=np.float64)

    ann_json = json.load(open(ann_path, "r")) #Ground Truth; ann_path= 'data/rescuenet/test_rescuenet.json'
    pred_json = result  # 预测结果; [name, score; name, score;...]  total batch size


    if matrix==True:
        predict = []
        target = []
        num = len(ann_json)
        for i in range(num):
            predict.append(pred_json[i]["scores"])
            target.append(ann_json[i]["target"])
        predict_binary = [[1 if prob >= 0.5 else 0 for prob in image_probs] for image_probs in
                              predict]

        cm = multilabel_confusion_matrix(target, predict_binary)
        labels = list(classes)

        num_classes = len(labels)
        cols = int(ceil(sqrt(num_classes)))
        rows = int(ceil(num_classes / cols))
        plt.figure(figsize=(cols * 4, rows * 4))
        for i in range(num_classes):
            ax = plt.subplot(rows, cols, i + 1)
            ax.matshow(cm[i], cmap=plt.cm.Blues, alpha=0.6)
            for x in range(cm[i].shape[0]):
                for y in range(cm[i].shape[1]):
                    plt.text(y, x, s=str(cm[i][x, y]), va='center', ha='center')
            plt.xticks([])
            plt.yticks([])
            plt.title(labels[i])
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        matrix_name = os.path.join(log_folder_path, f"matrix_{modelname}_{types}_{current_time}.png")
        plt.tight_layout()
        plt.savefig(matrix_name)
        plt.show()

    for i, _ in enumerate(tqdm(classes)):
        ap = json_map(i, pred_json, ann_json, types)
        aps[i] = ap

    precision_per_class, recall_per_class, f1_per_class = per_class_metric(pred_json, ann_json,num_classes)
    OP, OR, OF1, CP, CR, CF1 = json_metric(pred_json, ann_json, len(classes), types)
    file_name = modelname + '_' + types + '_log.txt'
    file_path = os.path.join(log_folder_path, file_name)
    with open(file_path, 'a') as file:
        file.write(f"Epoch: {epoch_now} of {modelname}\n")
        file.write("mAP: {:.2f}\n".format(np.mean(aps) * 100))
        file.write("CP: {:.2f}, CR: {:.2f}, CF1: {:.2f}\n".format(CP * 100, CR * 100, CF1 * 100))
        file.write("OP: {:.2f}, OR: {:.2f}, OF1: {:.2f}\n".format(OP * 100, OR * 100, OF1 * 100))
        file.write("\nLabel Specific Performance:\n")
        for i, class_name in enumerate(classes):
            file.write(
                f"{class_name} - AP: {aps[i] * 100:.2f}%, Precision: {precision_per_class[i] * 100:.2f}%, Recall: {recall_per_class[i] * 100:.2f}%, F1-score: {f1_per_class[i] * 100:.2f}%\n")
        file.write("-" * 40 + "\n")

    mAP=np.mean(aps)

    print("mAP: {:.2f}".format(mAP * 100))
    print("CP: {:.2f}, CR: {:.2f}, CF1: {:.2f}".format(CP * 100, CR * 100, CF1 * 100))
    print("OP: {:.2f}, OR: {:.2f}, OF1: {:.2f}".format(OP * 100, OR * 100, OF1 * 100))

    return mAP, CP, CR, CF1, OP, OR, OF1



