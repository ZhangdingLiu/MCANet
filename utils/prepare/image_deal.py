
#####################################  task 1

# from PIL import Image
# import numpy as np
#
# image_path = r'D:\Data\hurricane\code\CSRA-master\Dataset\rescuenet\segmentation-testset\test-label-img\10955_lab.png'
#
# annotation_image = Image.open(image_path)
# annotation_array = np.array(annotation_image)
#
# unique_classes = np.unique(annotation_array)
#
# colors = {
#     0: (255, 255, 255),  # Background - White
#     1: (0, 0, 255),      # Water - Blue
#     2: (128, 128, 128),  # Building_No_Damage - Gray
#     3: (255, 255, 0),    # Building_Minor_Damage - Yellow
#     4: (255, 165, 0),    # Building_Major_Damage - Orange
#     5: (255, 0, 0),      # Building_Total_Destruction - Red
#     6: (64, 64, 64),     # Vehicle - Dark Gray
#     7: (0, 255, 0),      # Road-Clear - Green
#     8: (0, 139, 139),    # Road-Blocked - Dark Cyan
#     9: (0, 128, 0),      # Tree - Dark Green
#     10: (0, 255, 255)    # Pool - Cyan
# }
#
# color_image_array = np.zeros((annotation_array.shape[0], annotation_array.shape[1], 3), dtype=np.uint8)
# for cls in unique_classes:
#     color_image_array[annotation_array == cls] = colors[cls]
#
# color_image = Image.fromarray(color_image_array)
# color_image.show()
#
# unique_classes, color_image
#
#
# # visualize_annotation('Dataset/rescuenet/segmentation-trainset/train-label-img/10778_lab.png')
#

#####################################  task 2
# import os
#
# data_path = 'Dataset/rescuenet/'
# anno_paths1 = os.path.join(data_path, "segmentation-trainset/train-org-img/")
# anno_paths2 = os.path.join(data_path, "segmentation-validationset/val-org-img/")
# anno_paths3 = os.path.join(data_path, "segmentation-testset/test-org-img/")
#
#
# with open('train.txt', 'w') as f:
#     for filename in os.listdir(anno_paths1):
#         if filename.endswith('.jpg'):
#
# print('File names saved to txt')
#
#
# with open('val.txt', 'w') as f:
#     for filename in os.listdir(anno_paths2):
#         if filename.endswith('.jpg'):
#
# print('File names saved to txt')
#
#
# with open('test.txt', 'w') as f:
#     for filename in os.listdir(anno_paths3):
#         if filename.endswith('.jpg'):
#
# print('File names saved to txt')

#####################################  task 3
# Find images where "Building_Total_Destruction" (index 4)
# and "Road-Blocked" (index 7) are present.
#
# import json
# import os
# import shutil
#
#
# def find_specific_images(json_file, output_dir):
#     # Load the JSON data
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#
#     # Define class indices based on cls_id0
#     building_total_destruction_index = 4  # "Building_Total_Destruction"
#     road_blocked_index = 7  # "Road-Blocked"
#     Water=0
#     Building_No_Damage=1
#     Building_Minor_Damage=2
#     Building_Major_Damage=3
#     Vehicle=5
#     Road_Clear=6
#     Tree=8
#     Pool=9
#
#     # Indices of building damage categories to exclude
#     exclude_indices = [2]  # "Building_No_Damage", "Building_Minor_Damage", "Building_Major_Damage"
#
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Find images that meet the criteria
#     filtered_images = []
#     for item in data:
#         target = item['target']
#         img_path = item['img_path']
#
#         # Check if "Building_Total_Destruction" and "Road-Blocked" are present
#         # if (target[building_total_destruction_index] == 1 and
#         #         target[road_blocked_index] == 1):
#
#         if (target[Building_Major_Damage] == 1):
#             # Ensure other building damage categories are not present
#             if all(target[idx] == 0 for idx in exclude_indices):
#                 filtered_images.append(img_path)
#
#     # Print all images that meet the criteria
#     print("Images that meet the criteria:")
#     for img in filtered_images:
#         print(img)
#
#     # Copy only the first 50 images to the output directory
#     for img in filtered_images[:50]:
#         shutil.copy(img, output_dir)
#
#     # Return the list of image paths
#     return filtered_images
#
#
# if __name__ == "__main__":
#     # Path to the trainval_rescuenet.json file
#     json_file = "data/rescuenet/trainval_rescuenet.json"
#     # Directory to save extracted images
#     output_dir = "demo_images/major damage"
#
#     # Find images and copy the first 20 to the output directory
#     images_with_conditions = find_specific_images(json_file, output_dir)
#
#     # Output the result
#     print(f"\nTotal images that meet the criteria: {len(images_with_conditions)}")
#     print(f"First 50 images copied to: {output_dir}")

#####################################  task 4
#   find major damage ;
# import json
# import os
# import shutil
# from collections import Counter
#
# def find_specific_images(json_file, output_dir):
#     # Load the JSON data
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#
#     # Define class indices and names based on cls_id0
#     labels = {
#         0: 'Water',
#         1: 'Building_No_Damage',
#         2: 'Building_Minor_Damage',
#         3: 'Building_Major_Damage',
#         4: 'Building_Total_Destruction',
#         5: 'Vehicle',
#         6: 'Road-Clear',
#         7: 'Road-Blocked',
#         8: 'Tree',
#         9: 'Pool'
#     }
#
#     Building_Major_Damage = 3
#
#     # Indices of building damage categories to exclude
#     exclude_indices = [2]  # "Building_Minor_Damage"
#
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Find images that meet the criteria
#     filtered_images = []
#     for item in data:
#         target = item['target']
#         img_path = item['img_path']
#
#         # Check if "Building_Major_Damage" is present
#         if target[Building_Major_Damage] == 1:
#             # Ensure the exclude categories are not present
#             if all(target[idx] == 0 for idx in exclude_indices):
#                 filtered_images.append((img_path, target))
#
#     # Print all images that meet the criteria along with their labels and counts
#     print("Images that meet the criteria:")
#     for img_path, target in filtered_images:
#         # Count labels in the image
#         label_counts = Counter()
#         for i, val in enumerate(target):
#             if val == 1:
#                 label_counts[labels[i]] += 1
#
#         # Sort labels by counts in ascending order
#         sorted_labels = sorted(label_counts.items(), key=lambda x: x[1])  # Sort in ascending order
#
#         # Print image path and sorted labels with counts
#         print(f"\nImage: {img_path}")
#         for label, count in sorted_labels:
#             print(f"  {label}: {count}")
#
#     # Copy only the first 100 images to the output directory
#     for img_path, _ in filtered_images[:100]:
#         shutil.copy(img_path, output_dir)
#
#     # Return the list of image paths
#     return filtered_images
#
# if __name__ == "__main__":
#     # Path to the trainval_rescuenet.json file
#     json_file = "data/rescuenet/trainval_rescuenet.json"
#     # Directory to save extracted images
#     output_dir = "demo_images/major_damage_sorted_order"
#
#     # Find images and copy the first 100 to the output directory
#     images_with_conditions = find_specific_images(json_file, output_dir)
#
#     # Output the result
#     print(f"\nTotal images that meet the criteria: {len(images_with_conditions)}")
#     print(f"First 100 images copied to: {output_dir}")

#####################################  task 4

import json
import os
import shutil
from collections import Counter

def find_specific_images(json_file, output_dir):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Define class indices and names based on cls_id0
    labels = {
        0: 'Water',
        1: 'Building_No_Damage',
        2: 'Building_Minor_Damage',
        3: 'Building_Major_Damage',
        4: 'Building_Total_Destruction',
        5: 'Vehicle',
        6: 'Road-Clear',
        7: 'Road_Blocked',
        8: 'Tree',
        9: 'Pool'
    }

    Building_Total_Destruction = 4
    Building_Major_Damage=3
    Road_Blocked=7

    # Indices of building damage categories to exclude
    exclude_indices = [0,1,2,4,5,9]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find images that meet the criteria
    filtered_images = []
    for item in data:
        target = item['target']
        img_path = item['img_path']

        # Check if "Building_Major_Damage" is present
        if target[Building_Major_Damage] == 1 :
            # Ensure the exclude categories are not present
            if all(target[idx] == 0 for idx in exclude_indices):
                # Calculate the total number of labels for sorting later
                label_count = sum(target)
                filtered_images.append((img_path, target, label_count))

    # Sort images by the total number of labels in ascending order
    filtered_images.sort(key=lambda x: x[2])  # Sort by label_count (the third element)

    # Print all images that meet the criteria along with their labels and counts
    print("Images sorted by the number of labels (from low to high):")
    for img_path, target, label_count in filtered_images:
        # Count labels in the image
        label_counts = Counter()
        for i, val in enumerate(target):
            if val == 1:
                label_counts[labels[i]] += 1

        # Print image path and sorted labels with counts
        print(f"\nImage: {img_path} (Total Labels: {label_count})")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

    # Copy only the first 100 images to the output directory
    for img_path, _, _ in filtered_images[:100]:
        shutil.copy(img_path, output_dir)

    # Return the list of image paths
    return filtered_images

if __name__ == "__main__":
    # Path to the trainval_rescuenet.json file
    json_file = "data/rescuenet/trainval_rescuenet.json"
    # Directory to save extracted images
    output_dir = "demo_images/majordamage_conference_paper"

    # Find images and copy the first 100 to the output directory
    images_with_conditions = find_specific_images(json_file, output_dir)

    # Output the result
    print(f"\nTotal images that meet the criteria: {len(images_with_conditions)}")
    print(f"all images copied to: {output_dir}")

