import os
import json
import numpy as np
from tqdm import tqdm
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont
import fiftyone as fo
from fiftyone import ViewField as F
import shutil
# from fiftyone.utils.coco import COCODetectionDatasetExporter

ground_cls_dict = {
    0 : "a dropped object", 
    1 : "a ground repair", 
    2 : "a white painting", 
    3 : "an arrow", 
}

def main():
    autra_dataset_dir = "/home/guorun.yang/data/autra"
    autra_image_dir = os.path.join(autra_dataset_dir, "images")
    autra_anno_pth = os.path.join(autra_dataset_dir, "annotations/drop_trainval.json")
    corner_dataset_dir = "/home/guorun.yang/data/cornercase"
    corner_image_dir = os.path.join(corner_dataset_dir, "images")
    corner_anno_pth = os.path.join(corner_dataset_dir, "annotations/trainval_4cls_rename.json")

    export_classes = [
        "dropped object",
        "vehicle",
        "person",
        "cone",
        "safety barrel",
        "ground repair",
        "white painting",
        "arrow",
    ]

    # Load the dataset
    autra_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=autra_image_dir,
        labels_path=autra_anno_pth,
    )
    
    corner_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=corner_image_dir,
        labels_path=corner_anno_pth,
    )
    corner_view = corner_dataset.filter_labels(
        "detections",
        F("label").is_in(["dropped object", "ground repair", "white painting", "arrow"])
    )
    autra_dataset.merge_samples(
        corner_view,
        fields = "detections",
    )
    # autra_session = fo.launch_app(autra_dataset)
    # autra_session.wait()

    export_pth = os.path.join(autra_dataset_dir, "annotations/annotations_merge.json")
    autra_dataset.persistent = True
    autra_dataset.export(
        # export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        labels_path = export_pth,
        label_field = "detections",
        classes=export_classes
    )
    # pretty print
    json_labels = {}
    with open(export_pth, 'r') as f:
        json_labels = json.load(f)
    with open(export_pth, 'w') as f:
        json.dump(json_labels, f, indent=4)

    # print("Corner case dataset: ", corner_dataset)
    # Filter the labels 


# def main():
#     drop_anno_pth = "/home/guorun.yang/data/autra/annotations/drop_trainval.json"
#     ground_anno_pth = "/home/guorun.yang/data/cornercase/annotations/trainval_4cls.json"
#     drop_label = {}
#     ground_label = {}
#     with open(drop_anno_pth, 'r') as f:
#         drop_label = json.load(f)
#     # Append ground cls to drop label
#     drop_label["categories"].append(
#         {
#             "id": 5,
#             "name": "ground pair",
#             "supercategory": null
#         }
#     )
#     drop_label["categories"].append(
#         {
#             "id": 6,
#             "name": "white painting",
#             "supercategory": null
#         }
#     )
#     drop_label["categories"].append(
#         {
#             "id": 7,
#             "name": "arrow",
#             "supercategory": null
#         },
#     )
#     drop_annos_num = len(drop_label["annotations"])

#     with open(ground_anno_pth, 'r') as f:
#         ground_label = json.load(f)
#     # ground_categories = ground_label["categories"]
#     ground_annos = ground_label["annotations"]
#     for i, lbl in enumerate(ground_annos):
#         cat_id = lbl["id"]
#         if cat_id == 1 or cat_id == 2 or cat_id == 3:

#     # pretty print
#     json_labels = {}
#     with open(export_pth, 'r') as f:
#         json_labels = json.load(f)
#     with open(export_pth, 'w') as f:
#         json.dump(json_labels, f, indent=4)


if __name__ == '__main__':
    main()
