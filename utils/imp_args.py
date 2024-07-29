import json
import copy

def _make_imp_args(ds_args):
    imp_args = copy.deepcopy(ds_args)
    if ds_args.task_name == 'long_term_forecast':
        imp_args.task_name ='imputation'
        imp_args.label_len = 0
        imp_args.pred_len = 0
        
        # 通过json文件获取填补模型参数
        with open(imp_args.imp_args_json, 'r') as f:
            data = json.load(f)
            config = data["config"]
            # 获取单独训练的填补模型权重
            weight_paths = data["weight_path"]

        # 读取参数
        for key, value in config.items():
            if hasattr(imp_args, key):
                setattr(imp_args, key, value)
        weight_path = weight_paths[str(imp_args.mask_rate)]
    
    return imp_args, weight_path