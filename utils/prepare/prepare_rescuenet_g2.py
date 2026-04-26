
import json

# original:
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
# }


# 将原来的标签 3 和 4 合并为 3: 'Severely damaged'，
# 2: 'Building_No_Damage'
# 3: 'Building_With_Damage'
# 5: 'Building_Collapsed'
# 7: 'Road-Clear'
# 8: 'Road-Blocked'

# 合并后的类别索引
keep_classes = [2, 3, 5, 7, 8]  # 最终的五个标签

def filter_and_merge_classes(json_file_path, output_file_path):
    with open(json_file_path, "r") as infile:
        data = json.load(infile)

    filtered_data = []

    for item in data:
        # 获取原始 target 数组
        target = item["target"]

        # 初始化新的 target，默认值为 0，长度为5，因为有5个最终类别
        new_target = [0] * len(keep_classes)

        # 保留 'Building_No_Damage'，对应类别 2
        new_target[0] = target[1]  # target[1] 是 Building_No_Damage

        # 合并 'Severely damaged'，合并原始类别 3 和 4
        if target[2] == 1 or target[3] == 1:
            new_target[1] = 1  # target[2] 或 target[3] 是 Severely damaged

        # 保留 'Building_Total_Destruction'，对应类别 5
        new_target[2] = target[4]  # target[4] 是 Building_Total_Destruction

        # 保留 'Road-Clear' 和 'Road-Blocked'
        new_target[3] = target[6]  # target[6] 是 Road-Clear
        new_target[4] = target[7]  # target[7] 是 Road-Blocked

        # 如果 new_target 数组中至少有一个类别为 1，保留该项
        if any(new_target):
            filtered_item = {
                "target": new_target,
                "img_path": item["img_path"]
            }
            filtered_data.append(filtered_item)

    # 将筛选后的数据写回新的 JSON 文件
    with open(output_file_path, "w") as outfile:
        json.dump(filtered_data, outfile, indent=4)
    print(f"Filtered and merged data saved to {output_file_path}")

# 输入和输出文件路径
trainval_json_path = "data/rescuenet/trainval_rescuenet.json"
test_json_path = "data/rescuenet/test_rescuenet.json"

filtered_trainval_json_path = "data/rescuenet/trainval_rescuenet_g2.json"
filtered_test_json_path = "data/rescuenet/test_rescuenet_g2.json"

# 执行过滤和合并操作
filter_and_merge_classes(trainval_json_path, filtered_trainval_json_path)
filter_and_merge_classes(test_json_path, filtered_test_json_path)
