
import json

#       original:
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

# 只保留 ：
# 2: 'Building_No_Damage'
# 3: 'Building_Minor_Damage'
# 4: 'Building_Major_Damage'
# 5: 'Building_Total_Destruction'
# 7: 'Road-Clear'
# 8: 'Road-Blocked'



# 要保留的类别索引 (1-based index)
keep_classes = [2, 3, 4, 5, 7, 8]

def filter_data(json_file_path, output_file_path):
    with open(json_file_path, "r") as infile:
        data = json.load(infile)

    filtered_data = []

    for item in data:
        # 获取原始 target 数组
        target = item["target"]
        # 根据要保留的类别索引，筛选出相应的 target
        new_target = [target[i - 1] for i in keep_classes]

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
    print(f"Filtered data saved to {output_file_path}")

# 输入和输出文件路径
trainval_json_path = "data/rescuenet/trainval_rescuenet.json"
test_json_path = "data/rescuenet/test_rescuenet.json"

filtered_trainval_json_path = "data/rescuenet/trainval_rescuenet_g3.json"
filtered_test_json_path = "data/rescuenet/test_rescuenet_g3.json"

# 执行过滤操作
filter_data(trainval_json_path, filtered_trainval_json_path)
filter_data(test_json_path, filtered_test_json_path)
