import os
import json

freq_cls = [
    # 'pedestrian', 
    # 'cyclist', 
    # 'car', 
    # 'truck', 
    # 'construction_vehicle', 
    # 'barrier', 
    # 'bollard', 
    # 'traffic_cone',
    'person',
    'bicycle',
    'car_(automobile)',
    'truck',
    'barrel',
    'pole',
    'cone',
]

common_cls = [
    # 'tricycle', 
    # 'bus', 
    # 'bicycle', 
    # 'moped', 
    # 'dog', 
    # 'traffic_sign', 
    # 'dustbin', 
    # 'misc',
    'tricycle',
    'bus_(vehicle)',
    'motorcycle',
    'dog',
    'street_sign',
    'trash_can',
]

rare_cls = [
    'motorcycle', 
    'stroller', 
    'cart', 
    'sentry_box', 
    'traffic_island', 
    'traffic_light', 
    'debris', 
    'suitcace', 
    'concrete_block', 
    'machinery', 
    'garbage', 
    'plastic_bag', 
    'stone'
]

cls_mapping_dict = {
    'pedestrian'                : 'person',
    'cyclist'                   : 'bicycle',
    'car'                       : 'car_(automobile)',
    'truck'                     : 'truck',
    'tricycle'                  : 'tricycle',
    'bus'                       : 'bus_(vehicle)',
    'bicycle'                   : 'bicycle',
    'moped'                     : 'motorcycle',
    'motorcycle'                : 'motorcycle',
    'stroller'                  : 'baby_buggy',
    'cart'                      : 'cart',
    'construction_vehicle'      : 'truck',                  # ?
    'dog'                       : 'dog',
    'barrier'                   : 'barrel',
    'bollard'                   : 'pole',
    'sentry_box'                : '',
    'traffic_cone'              : 'cone',
    'traffic_island'            : '',
    'traffic_light'             : 'traffic_light',
    'traffic_sign'              : 'street_sign',
    'debris'                    : '',
    'suitcace'                  : 'suitcase',
    'dustbin'                   : 'trash_can',
    'concrete_block'            : '',
    'machinery'                 : '',
    'garbage'                   : 'garbage',
    'plastic_bag'               : 'plastic_bag',
    'stone'                     : '',
    'misc'                      : '',
}


if __name__ == '__main__':
    coda_json_dir = "/home/guorun.yang/data/CODA/annotations"
    lvis_json_pth = "/home/guorun.yang/data/lvis/annotations/lvis_v1_val.json"
    src_json_pth = os.path.join(coda_json_dir, "conda_val.json")
    des_json_pth = os.path.join(coda_json_dir, "conda2lvis_val.json")
    print("Src json pth: ", src_json_pth)
    des_label = {
        'categories' : [],
        'annotations' : [],
        'images' : [],
    }
    coda2lvis_cat = {}
    coda_cls_cnt = {}
    lvis_cls2id, lvis_id2cls = {}, {}
    coda_cls2id, coda_id2cls = {}, {}
    lvis_label = {}
    
    # Save the coda categories that occur in LVIS
    coda_cls_set = set()
    for src_name, des_name in cls_mapping_dict.items():
        coda_cls_cnt[src_name] = 0
        if des_name == '':
            continue
        if des_name not in coda_cls_set:
            coda_cls_set.add(des_name)

    # Build the dict of lvis cls: cat_name -> cat_id
    with open(lvis_json_pth, 'r') as f:
        lvis_label = json.load(f)
        lvis_cat = lvis_label['categories']
        for cat_info in lvis_cat:
            cat_cls = cat_info['name']
            cat_id = cat_info['id']
            lvis_cls2id[cat_cls] = cat_id
            lvis_id2cls[cat_id] = cat_cls

    with open(src_json_pth, 'r') as f:
        coda_label = json.load(f)
        des_label['images'] = coda_label['images']

        coda_label_cat = coda_label['categories']
        lvis_label_cat = lvis_label['categories']

        # Build the dict from CODA category ID -> LVIS category ID
        for coda_cat_info in coda_label_cat:
            coda_cat_name = coda_cat_info['name']
            coda_cat_id = coda_cat_info['id']
            coda_cls2id[coda_cat_name] = coda_cat_id
            coda_id2cls[coda_cat_id] = coda_cat_name

            lvis_mapping_name = cls_mapping_dict[coda_cat_name]
            if lvis_mapping_name != '':
                lvis_mapping_id = lvis_cls2id[lvis_mapping_name]
                coda2lvis_cat[coda_cat_id] = lvis_mapping_id
            
        # Transform the category from CODA to LVIS
        for lvis_cat_info in lvis_label_cat:
            lvis_cat_name = lvis_cat_info['name']
            lvis_cat_id = lvis_cat_info['id']
            lvis_cat_freq = lvis_cat_info['frequency']
            if lvis_cat_name in freq_cls:
                lvis_cat_freq = 'f'
            elif lvis_cat_name in common_cls:
                lvis_cat_freq = 'c'
            else:
                lvis_cat_freq = 'r'
            
            des_cat_info = {
                'id' : lvis_cat_id,
                'name' : lvis_cat_name,
                'frequency' : lvis_cat_freq,
            }
            des_label['categories'].append(des_cat_info)

        # Transform the annotations from CODA to LVIS
        coda_label_annos = coda_label['annotations']
        for i, lbl in enumerate(coda_label_annos):
            anno_cls_id = lbl['category_id']
            anno_cls_name = coda_id2cls[anno_cls_id]
            coda_cls_cnt[anno_cls_name] += 1
            lbl_lvis_name = cls_mapping_dict[anno_cls_name]
            if lbl_lvis_name == '':
                continue
            # lvis_cls_name = cls_mapping_dict[anno_cls_name]
            # lvis_cls_id = lvis_cls_dict[lvis_cls_name]
            lbl_lvis_id = coda2lvis_cat[anno_cls_id]
            lbl_bbox = lbl['bbox']
            bbox_area = lbl_bbox[2] * lbl_bbox[3]
            lbl['category_id'] = lbl_lvis_id
            lbl['category_name'] = lbl_lvis_name
            lbl['area'] = bbox_area
            lbl['segmentation'] = [[]]
            des_label['annotations'].append(lbl)
            
            # print("Src Cls ID {} -> {} -> {} -> {}".format(
            #     anno_cls_id, anno_cls_name, lvis_cls_name, lvis_cls_id
            # ))
            # if i > 100:
            #     break
            # Accumulate the cls count

        # Supplement image info
        for n, image_dict in enumerate(des_label['images']):
            image_dict['neg_category_ids'] = []
            image_dict['not_exhaustive_category_ids'] = []

    # with open(des_json_pth, 'w') as f:
    #     json.dump(des_label, f)

    # Determine the frequency, common, 
    freq_cls, common_cls, rare_cls = [], [], []
    for cls_name, cls_num in coda_cls_cnt.items():
        print("Cls name: {}, number: {}".format(cls_name, cls_num))
        if cls_num >= 1000:
            freq_cls.append(cls_name)
        elif cls_num >= 100:
            common_cls.append(cls_name)
        else:
            rare_cls.append(cls_name)
    print("Freq cls:   ", freq_cls)
    print("Common cls: ", common_cls)
    print("Rare cls:   ", rare_cls)
            

    
        # des_label['images'] = src_label['images']
        # src_label_cat = src_label['categories']
        # des_label_cat = []
        # for cat_name in src_label_cat:

