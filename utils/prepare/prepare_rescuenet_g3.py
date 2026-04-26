
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

# 2: 'Building_No_Damage'
# 3: 'Building_Minor_Damage'
# 4: 'Building_Major_Damage'
# 5: 'Building_Total_Destruction'
# 7: 'Road-Clear'
# 8: 'Road-Blocked'

keep_classes = [2, 3, 4, 5, 7, 8]

def filter_data(json_file_path, output_file_path):
    with open(json_file_path, "r") as infile:
        data = json.load(infile)

    filtered_data = []

    for item in data:
        target = item["target"]
        new_target = [target[i - 1] for i in keep_classes]

        if any(new_target):
            filtered_item = {
                "target": new_target,
                "img_path": item["img_path"]
            }
            filtered_data.append(filtered_item)

    with open(output_file_path, "w") as outfile:
        json.dump(filtered_data, outfile, indent=4)
    print(f"Filtered data saved to {output_file_path}")

trainval_json_path = "data/rescuenet/trainval_rescuenet.json"
test_json_path = "data/rescuenet/test_rescuenet.json"

filtered_trainval_json_path = "data/rescuenet/trainval_rescuenet_g3.json"
filtered_test_json_path = "data/rescuenet/test_rescuenet_g3.json"

filter_data(trainval_json_path, filtered_trainval_json_path)
filter_data(test_json_path, filtered_test_json_path)
