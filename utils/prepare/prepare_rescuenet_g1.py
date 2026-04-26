
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

# 把类别3,4,5 合并当作 Building_Collapsed
# 2: 'Building_No_Damage'
# 3: 'Building_With_Damage'
# 7: 'Road-Clear'
# 8: 'Road-Blocked'

# 合并后的类别索引
keep_classes = [2, 3, 7, 8]

def filter_and_merge_classes(json_file_path, output_file_path):
    with open(json_file_path, "r") as infile:
        data = json.load(infile)

    filtered_data = []

    for item in data:
        # 获取原始 target 数组
        target = item["target"]

        # 初始化新的 target，默认值为0，长度为4，因为有4个最终类别
        new_target = [0] * len(keep_classes)

        # 处理 'Building_No_Damage'
        new_target[0] = target[1]  # 类别2对应Building_No_Damage

        # 合并 'Building_Collapsed'（合并3, 4, 5的标签）
        if target[2] == 1 or target[3] == 1 or target[4] == 1:
            new_target[1] = 1  # 类别3是Building_Collapsed

        # 处理 'Road-Clear' 和 'Road-Blocked'
        new_target[2] = target[6]  # 类别7对应Road-Clear
        new_target[3] = target[7]  # 类别8对应Road-Blocked

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

filtered_trainval_json_path = "data/rescuenet/trainval_rescuenet_g1.json"
filtered_test_json_path = "data/rescuenet/test_rescuenet_g1.json"

# 执行过滤和合并操作
filter_and_merge_classes(trainval_json_path, filtered_trainval_json_path)
filter_and_merge_classes(test_json_path, filtered_test_json_path)