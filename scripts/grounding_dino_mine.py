# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import time
import numpy as np
import requests

from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from typing import Dict, List, Optional, TextIO, Tuple

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    # parser.add_argument('--texts', 
    #     default=" there is a dropped object on the road . an arrow . a white painting . a ground repair . a cone  ",
    #     type=str,              
    #     help='text prompt')
    parser.add_argument('--texts', 
        default=" there is a dropped object on the road . \
            vehicle . cone . person . safety barrel . ground repair . \
            white painting . arrow ",
        type=str,              
        help='text prompt')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--tag-score-thr',
        type=float,
        default=0.7,
        help='bbox score threshold for tag')
    parser.add_argument("--is_dev",
        action='store_true',
        help="database is prod or not.")
    parser.add_argument(
        '--batch-size', type=int, default=32, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='random',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    # Collect init args
    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)
    
    # Collect data args
    data_kws = ['tag_score_thr', 'is_dev']
    data_args = {}
    for data_kw in data_kws:
        data_args[data_kw] = call_args.pop(data_kw)

    return init_args, call_args, data_args


def _save_dino_tags(
        record_path: str, 
        vehicle_id: str, 
        is_dev: bool, 
        dino_tags: list):
    json_payload = {
        'record_path': record_path,
        'vehicle_id': vehicle_id,
        'frame_tags': dino_tags
    }
    print(f'json_payload: {json_payload} is_dev: {is_dev}')
    if is_dev:
        url = 'http://ram-tag-index-service-dev.autra.tech/write'
    else:
        url = 'http://ram-tag-index-service.autra.tech/write'
    # print("URL: ", url)
    response = requests.post(
        url=url,
        json=json_payload,
        timeout=3
    )
    json_obj = response.json()
    response.close()
    print(f'json_obj: {json_obj}')


def _format_record_path(record_path: str):
    img_dir = os.path.join(
        record_path, 
        '_apollo_sensor_camera_upmiddle_left_30h_image_compressed'
    )
    if os.path.isdir(img_dir):
        record_name = record_path.split('/')[-1]
        vehicle_id = record_name.split('_')[0]
        return vehicle_id, record_name, img_dir
    else:
        return 'unknown', 'unknown', record_path


def _generate_dino_tags(
        img_root: str,
        dino_result : Dict,
        score_thre : 0.7):
    img2time = {}
    format_tags = []
    total_frames, drop_frames = 0, 0
    if not os.path.exists(os.path.join(img_root, 'timestamps')):
        return format_tags, total_frames, drop_frames
    # Load the frames from timestamps
    with open(os.path.join(img_root, 'timestamps'), 'r', encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.split(' ')
            timestamp = round(float(tokens[1]) * 1000)
            img2time[tokens[0]] = timestamp
    
    # Load the image paths from results
    img_list = [frame_input["img_path"] for frame_input in dino_result["inputs"]]
    total_frames = len(img_list)
    for i, frame_pred in enumerate(dino_result["predictions"]):
        frame_labels = np.array(frame_pred["labels"])
        frame_bboxes = np.array(frame_pred["bboxes"])
        frame_scores = np.array(frame_pred["scores"])
        valid_indices = np.where(
            (np.logical_and(frame_labels == 0, frame_scores >= score_thre))
        )
        frame_dino_tags = []
        if len(valid_indices[0]) > 0:
            drop_frames += 1
            frame_dino_tags = ["dino_dropped"]
            img_path = img_list[i]
            img_name = (img_path.split('/'))[-1].split('.')[0]
            if img_name in img2time:
                format_tags.append({
                    'timestamp': img2time[img_name],
                    # 'image_name' : img_name,
                    'tags': frame_dino_tags
                })
        # frame_dino_tags = {}
        # if len(valid_indices[0]) > 0:
        #     drop_bboxes = frame_bboxes[valid_indices].tolist()
        #     drop_scores = frame_scores[valid_indices].tolist()
        #     frame_dino_tags['bboxes'] = drop_bboxes
        #     frame_dino_tags['scores'] = drop_scores
        # else:
        #     frame_dino_tags['bboxes'] = []
        #     frame_dino_tags['scores'] = []
        # img_path = img_list[i]
        # img_name = (img_path.split('/'))[-1].split('.')[0]
        # if img_name in img2time:
        #     format_tags.append({
        #         'timestamp': img2time[img_name],
        #         'image_name' : img_name,
        #         'tags': frame_dino_tags
        #     })
    return format_tags, total_frames, drop_frames


def main():
    init_args, call_args, data_args = parse_args()
    # parse the record info
    vehicle_id, record_name, img_dir = _format_record_path(call_args["inputs"])
    print("Vehicle ID: ", vehicle_id, " Record name: ", record_name, " Image dir: ", img_dir)
    call_args["inputs"] = img_dir
    print("Inputs: ", call_args["inputs"])
    
    inferencer = DetInferencer(**init_args)
    start = time.time()
    result_dict = inferencer(**call_args)
    end = time.time()
    print("Infer time: ", end-start)
    # print("Result dict: ", result_dict.keys())
    # print("Result input len: ", len(result_dict["inputs"]))
    # print("Result predictions len: ", len(result_dict["predictions"]))
    if os.path.isdir(call_args["inputs"]):
        dino_tags, total_frames, drop_frames = _generate_dino_tags(
            img_root=call_args["inputs"], 
            dino_result=result_dict, 
            score_thre = data_args['tag_score_thr']
        )
        if vehicle_id != 'unknown' and record_name != 'unknown':
            # tmp_pth = "./result.json"
            # with open(tmp_pth, 'w') as f:
            #     json.dump(dino_tags, f, indent=4)
            normal_record_path = os.path.join('yizhuang/raw_records', record_name)
            _save_dino_tags(
                record_path=normal_record_path, 
                vehicle_id=vehicle_id,
                is_dev=data_args['is_dev'], 
                dino_tags=dino_tags
            )
        print_log(f'Total frame: {total_frames}, Drop object frames: {drop_frames}')
    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()
