import torch
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    table1 = PrettyTable(["Modules-Level1", "Parameters"])
    table2 = PrettyTable(["Modules-Level2", "Parameters"])
    module_l1_num = {}
    module_l2_num = {}

    total_params = 0
    total_params1 = 0
    total_params2 = 0
    if "state_dict" in model.keys():
        model = model["state_dict"]
    # for name, parameter in model.named_parameters():
    for name, parameter in model.items():
        # if not parameter.requires_grad:
        #     continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        if "." in name:
            name_level1 = name.split(".")[0]
            name_level2 = name.split(".")[0] + "." + name.split(".")[1]
            if name_level1 in module_l1_num:
                module_l1_num[name_level1] += params
            else:
                module_l1_num[name_level1] = params

            if name_level2 in module_l2_num:
                module_l2_num[name_level2] += params
            else:
                module_l2_num[name_level2] = params
        else:
            if name in module_l1_num:
                module_l1_num[name] += params
            else:
                module_l1_num[name] = params

            if name in module_l2_num:
                module_l2_num[name] += params
            else:
                module_l2_num[name] = params

    for module_name, module_num in module_l1_num.items():
        table1.add_row([module_name, module_num])

    for module_name, module_num in module_l2_num.items():
        table2.add_row([module_name, module_num])

    print(table)
    print(table1)
    print(table2)
    print(f"Total Params: {total_params}")
    return total_params

if __name__ == '__main__':
    # model_pth = "/work01/guorun/mmdetection/weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
    # model_pth = "/work01/guorun/mmdetection/weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
    model_pth = "/work01/guorun/mmdetection/work_dirs/grounding_dino_swin-b_finetune_4xb2_10e_autra_cornercase_trainval/best_coco_drop_epoch_19.pth"
    model = torch.load(model_pth)
    count_parameters(model)