import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mealpy.swarm_based.SSA import BaseSSA
import math
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# ===================== 数据读取与预处理 =====================
data = pd.read_excel('file path', sheet_name="sheet_name")
feat_cols = ['Geomechanical Parameters', 'Geometric Parameters', 'Tunneling Parameters']
label_col = ['settlement']
data = data.astype(np.float32)

# 数据归一化
scaler_feat = MinMaxScaler(feature_range=(-1, 1))
scaler_label = MinMaxScaler(feature_range=(-1, 1))

data[feat_cols] = scaler_feat.fit_transform(data[feat_cols].values)
data[label_col] = scaler_label.fit_transform(data[label_col].values)

# ===================== 划分训练/测试集 =====================
X = data[feat_cols].values
y = data[label_col].values
X_train = X[:400]
y_train = y[:400]
X_test = X[400:450]
y_test = y[400:450]


# ===================== 定义优化目标函数 =====================
def opt_func(OPT):
    n_estimators = max(1, math.ceil(OPT[0]))  # 决策树数量
    max_depth = max(1, math.ceil(OPT[1]))  # 树最大深度

    # 初始化随机森林回归器
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    # 训练模型
    rf.fit(X_train, y_train.ravel())
    y_train_pred = rf.predict(X_train)

    # 反归一化（还原真实尺度）
    y_true = scaler_label.inverse_transform(y_train)
    y_pred = scaler_label.inverse_transform(y_train_pred.reshape(-1, 1))

    mse = mean_squared_error(y_true, y_pred)
    print(f'树数量: {n_estimators}, 树深度: {max_depth}, 训练集MSE: {mse:.4f}')
    return mse


# ===================== SSA优化参数设置 =====================
# 优化参数边界：[树数量, 树深度]
low_bound = [10, 3]  # 树数量最小10，深度最小3
up_bound = [200, 50]  # 树数量最大200，深度最大50

problem_dict = {
    "fit_func": opt_func,
    "lb": low_bound,
    "ub": up_bound,
    "minmax": "min"
}

# SSA算法参数
epoch = 50  # 迭代次数
pop_size = 30  # 种群大小
ST = 0.8  # 选择阈值
PD = 0.2  # 探测概率
SD = 0.1  # 搜索概率

# ===================== 运行SSA优化 =====================
model_ssa = BaseSSA(epoch, pop_size, ST, PD, SD)
best_params, best_loss = model_ssa.solve(problem_dict)

# 解析最优参数
best_n_estimators = max(1, math.ceil(best_params[0]))
best_max_depth = max(1, math.ceil(best_params[1]))

# ===================== 输出最优结果 =====================
print('\n==================== 最优参数 ====================')
print(f'最优决策树数量: {best_n_estimators}')
print(f'最优树最大深度: {best_max_depth}')
print(f'训练集最优MSE: {best_loss:.4f}')

# 用最优参数训练最终模型
final_rf = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth)
final_rf.fit(X_train, y_train.ravel())