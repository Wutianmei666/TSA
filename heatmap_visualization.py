import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset_list = ['ETTm1', 'ETTh1', 'ECL', 'Weather']
method_list = ['mean', 'nearest', 'linear', 'TimesNet']
mask_rate_list = ['0.125', '0.25', '0.375', '0.5', '0.625', '0.75']

array_list = []
for method in method_list:
    path = 'count_imp_loss/ETTm1_96_0.5/'+method+'/'+'mse.npy'
    array_list.append(np.load(path).reshape(-1,48))

heatmap_array = np.concatenate(array_list)

# 调整热图的比例
plt.figure(figsize=(48, 4))  # 设置横纵比

# 生成热图，cbar=False 去掉 colorbar，如果需要可以去掉这个参数
ax = sns.heatmap(heatmap_array, annot=True, fmt=".3f", cbar=True, 
                 xticklabels=range(1, 49), yticklabels=method_list)

plt.xticks(rotation=0)  # 保持横轴标签水平显示
plt.show()
