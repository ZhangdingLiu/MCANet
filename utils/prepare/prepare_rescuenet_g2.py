
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

# 2: 'Building_No_Damage'
# 3: 'Building_With_Damage'
# 5: 'Building_Collapsed'
# 7: 'Road-Clear'
# 8: 'Road-Blocked'

keep_classes = [2, 3, 5, 7, 8]

def filter_and_merge_classes(json_file_path, output_file_path):
    with open(json_file_path, "r") as infile:
        data = json.load(infile)

    filtered_data = []

    for item in data:
        target = item["target"]

        new_target = [0] * len(keep_classes)

        new_target[0] = target[1]  # target[1]  Building_No_Damage

        if target[2] == 1 or target[3] == 1:
            new_target[1] = 1  # target[2]  target[3]  Severely damaged

        new_target[2] = target[4]  # target[4]  Building_Total_Destruction

        new_target[3] = target[6]  # target[6]  Road-Clear
        new_target[4] = target[7]  # target[7]  Road-Blocked

        if any(new_target):
            filtered_item = {
                "target": new_target,
                "img_path": item["img_path"]
            }
            filtered_data.append(filtered_item)

    with open(output_file_path, "w") as outfile:
        json.dump(filtered_data, outfile, indent=4)
    print(f"Filtered and merged data saved to {output_file_path}")

trainval_json_path = "data/rescuenet/trainval_rescuenet.json"
test_json_path = "data/rescuenet/test_rescuenet.json"

filtered_trainval_json_path = "data/rescuenet/trainval_rescuenet_g2.json"
filtered_test_json_path = "data/rescuenet/test_rescuenet_g2.json"

filter_and_merge_classes(trainval_json_path, filtered_trainval_json_path)
filter_and_merge_classes(test_json_path, filtered_test_json_path)
