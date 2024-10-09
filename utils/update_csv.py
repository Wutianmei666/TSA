import pandas as pd
import numpy as np

def update_csv(df, new_data):
    # 解包新的数据
    new_seed = new_data["随机种子"]
    new_dataset = new_data["数据集"]
    new_imp_model = new_data["填补模型"]
    new_downstream_model = new_data["下游模型"]
    new_mask_rate = new_data["掩码率"]

    """随机种子、数据集、填补模型和掩码一定时，无论是何上游模型，他们的常规填补方法损失一样"""
    row = df.loc[(df['随机种子']==new_seed) &
                  (df['数据集']==new_dataset)&
                  (df['填补模型']==new_downstream_model)&
                  (df['下游模型']==new_downstream_model)&
                  (df['掩码率']==new_mask_rate)]
    
    if row.empty == False :
        typical_padding = ['均值','最近邻','线性']
        for type in typical_padding:
            if type not in new_data :
                new_data[type] = row[type]


    # 判断有无对应种子，无则追加
    if new_seed not in df['随机种子'].values:
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        return df

    # 判断有无对应的数据集，无则追加
    seed_rows = df[df['随机种子'] == new_seed]
    if new_dataset not in seed_rows['数据集'].values:
        next_seed_start = seed_rows.index[-1] + 1
        df = pd.concat([df.iloc[:next_seed_start], pd.DataFrame([new_data]), df.iloc[next_seed_start:]], ignore_index=True)
        return df

    # 判断有无对应的填补模型，无则追加
    dataset_rows = seed_rows[seed_rows['数据集'] == new_dataset]
    if new_imp_model not in dataset_rows['填补模型'].values:
        next_dataset_start = dataset_rows.index[-1] + 1
        # 插入新数据在数据集的最后
        df = pd.concat([df.iloc[:next_dataset_start], pd.DataFrame([new_data]), df.iloc[next_dataset_start:]], ignore_index=True)
        return df
    
    # 判断有无对应的下游模型，无则追加
    imp_model_rows = dataset_rows[dataset_rows['填补模型'] == new_imp_model]
    if new_downstream_model not in imp_model_rows['下游模型'].values:
        last_fill_model_index = imp_model_rows.index[-1] + 1
        df = pd.concat([df.iloc[:last_fill_model_index], pd.DataFrame([new_data]), df.iloc[last_fill_model_index:]], ignore_index=True)
        return df

    # 判断有无对应的掩码率
    downstream_rows = imp_model_rows[imp_model_rows['下游模型'] == new_downstream_model]
    mask_rates = downstream_rows['掩码率'].apply(lambda x: float(x.strip('%')))
    new_mask_rate_value = float(new_mask_rate.strip('%'))

    # 如果掩码率存在
    if new_mask_rate_value in mask_rates.values:
        # 获取需要更新的行索引
        index_to_update = downstream_rows[mask_rates == new_mask_rate_value].index[0]

        # 更新数据：仅更新 new_data 中存在的列，保留其他列
        for col in new_data:
            df.at[index_to_update, col] = new_data[col]
    else:
        # 插入数据，按掩码率排序
        insertion_index = mask_rates.searchsorted(new_mask_rate_value)+mask_rates.index[0]
        df = pd.concat([df.iloc[:insertion_index], pd.DataFrame([new_data]), df.iloc[insertion_index:]]).reset_index(drop=True)

    return df
