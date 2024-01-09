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
    # autra_anno_pth = os.path.join(autra_dataset_dir, "annotations/drop_train.json")
    autra_anno_pth = os.path.join(autra_dataset_dir, "annotations_new/trainval_no_vehicle_cls.json")

    corner_dataset_dir = "/home/guorun.yang/data/cornercase"
    corner_image_dir = os.path.join(corner_dataset_dir, "images")
    corner_anno_pth = os.path.join(corner_dataset_dir, "annotations/trainval_4cls_rename.json")

    export_classes = [
        "dropped object",
        # "vehicle",
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
    # autra_view = autra_dataset.filter_labels(
    #     "detections",
    #     F("label").is_in(export_classes)
    # )
    autra_filter_dataset = (
        autra_dataset.select_fields("detections").filter_labels(
            "detections",
            F("label").is_in(export_classes)
        )
    ).clone()
    corner_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=corner_image_dir,
        labels_path=corner_anno_pth,
    )
    # corner_view = corner_dataset.match("file_name").starts_with("A001")
    corner_view = corner_dataset.filter_labels(
        "detections",
        F("label").is_in(export_classes)
    )
    corner_view = corner_dataset.match(
        F("filepath").starts_with("/home/guorun.yang/data/cornercase/images/A001")
    )
    # corner_session = fo.launch_app(corner_view)
    # corner_session.wait()

    autra_filter_dataset.merge_samples(
        corner_view,
        fields = "detections",
    )

    export_pth = os.path.join(autra_dataset_dir, "annotations/drop_trainval_merge_no_vehicle_1226.json")
    autra_filter_dataset.persistent = True
    autra_filter_dataset.export(
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


# def check_duplicate_image(image_dir):
#     image_list = sorted(os.listdir(image_dir))
#     autra_frame_set = set()
#     cornercase_image_list = []
#     for image_fn in image_list:
#         if image_fn.startswith("16"):
#             frame_name = image_fn.rsplit(".", 1)[0]
#             autra_frame_set.add(frame_name)
#         else:
#             cornercase_image_list.append(image_fn)
#     print()
#     print("Cornercase image list: ", cornercase_image_list)


if __name__ == '__main__':
    main()

