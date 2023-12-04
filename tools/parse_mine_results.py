import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, help='Base directory')
    parser.add_argument('--pred', type=str, default='preds', help='Predict directory')
    parser.add_argument('--vis', type=str, default='vis', help='Vis directory')
    parser.add_argument('--drop_pred', type=str, default='drop_pred', help='Predict drop object directory')
    parser.add_argument('--drop_vis', type=str, default='drop_vis', help='Vis drop object directory')
    parser.add_argument('-s', '--score', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    pred_dir = os.path.join(args.base_dir, args.pred)
    vis_dir = os.path.join(args.base_dir, args.vis)
    drop_pred_dir = os.path.join(args.base_dir, args.drop_pred)
    drop_vis_dir = os.path.join(args.base_dir, args.drop_vis)
    os.makedirs(drop_pred_dir, exist_ok=True)
    os.makedirs(drop_vis_dir, exist_ok=True)
    pred_list = sorted(os.listdir(pred_dir))
    for i, pred_fn in enumerate(tqdm(pred_list)):
        frame_name = pred_fn.rsplit('.', 1)[0]
        src_pred_path = os.path.join(pred_dir, pred_fn)
        src_vis_path = os.path.join(vis_dir, frame_name + '.jpg')
        with open(src_pred_path, 'r') as f:
            frame_dets = json.load(f)
            frame_drop_dets = {
                'bboxes' : [],
                'scores' : [],
            }
            frame_dets_labels = np.array(frame_dets['labels'])
            frame_dets_scores = np.array(frame_dets['scores'])
            frame_dets_bboxes = np.array(frame_dets['bboxes'])
            valid_indices = np.where(
                (np.logical_and(frame_dets_labels == 0, frame_dets_scores >= args.score))
            )
            # print("Valid indices: ", valid_indices[0])
            # print("valid indices: ", len(valid_indices[0]))
            if len(valid_indices[0]) > 0:
                des_pred_path = os.path.join(drop_pred_dir, pred_fn)
                des_vis_path = os.path.join(drop_vis_dir, frame_name + '.jpg')
                drop_bboxes = frame_dets_bboxes[valid_indices].tolist()
                drop_scores = frame_dets_scores[valid_indices].tolist()
                frame_drop_dets['bboxes'] = drop_bboxes
                frame_drop_dets['scores'] = drop_scores
                shutil.copyfile(src_vis_path, des_vis_path)
                with open(des_pred_path, 'w') as f:
                    json.dump(frame_drop_dets, f, indent = 4)
            

