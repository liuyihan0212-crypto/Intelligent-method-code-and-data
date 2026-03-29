import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mealpy.swarm_based.SSA import BaseSSA
import math
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# ===================== 1. 解析接口=====================
def analytical_settlement(geo_mech_param, geo_geom_param, tunneling_param):
    settlement = 0.5
    # 沉降值需要根据参数以及解析公式具体计算，此处的0.5为占位示例
    # 具体值可通过Analytical solution-displacement.m 文件计算
    return settlement


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

# ===================== 权重参数设置 =====================
weight_data = 0.7    # 数据损失权重
weight_analytical = 0.3  # 解析损失权重


# ===================== 2. 定义优化目标函数（含解析损失） =====================
def opt_func(OPT):
    # 解析优化参数（决策树数量、最大深度，确保为正整数）
    n_estimators = max(1, math.ceil(OPT[0]))
    max_depth = max(1, math.ceil(OPT[1]))

    # 初始化并训练随机森林模型
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X_train, y_train.ravel())
    y_train_pred = rf.predict(X_train)

    # -------- 计算数据损失：RF预测值 vs 真实标签 --------
    y_true = scaler_label.inverse_transform(y_train)  # 真实值
    y_pred = scaler_label.inverse_transform(y_train_pred.reshape(-1, 1))  # 预测值
    data_mse = mean_squared_error(y_true, y_pred)

    # -------- 计算解析损失：RF预测值 vs 解析解 --------
    # 1. 生成所有训练样本的解析解
    analytical_pred = []
    for sample in X_train:
        # 提取单个样本的三个特征参数
        geo_mech = sample[0]
        geo_geom = sample[1]
        tunneling = sample[2]
        ana_sett = analytical_settlement(geo_mech, geo_geom, tunneling)
        analytical_pred.append(ana_sett)
    analytical_pred = np.array(analytical_pred)
    analytical_mse = mean_squared_error(y_train_pred, analytical_pred)

    # -------- 总损失 = 权重*数据损失 + 权重*解析损失 --------
    total_loss = weight_data * data_mse + weight_analytical * analytical_mse

    # 打印中间结果（便于调试优化过程）
    print(
        f'树数量: {n_estimators}, 树深度: {max_depth}, 数据MSE: {data_mse:.4f}, 解析MSE: {analytical_mse:.4f}, '
        f'加权数据损失: {weight_data * data_mse:.4f}, 加权解析损失: {weight_analytical * analytical_mse:.4f}, 总MSE: {total_loss:.4f}')
    return total_loss


# ===================== SSA优化参数设置 =====================
# 优化参数边界：[树数量, 树深度]
low_bound = [10, 3]  # 树数量最小10，深度最小3
up_bound = [200, 50]  # 树数量最大200，深度最大50

problem_dict = {
    "fit_func": opt_func,
    "lb": low_bound,
    "ub": up_bound,
    "minmax": "min"  # 优化目标：最小化总损失
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

# ===================== 输出最优结果 ====================
print('\n==================== 最优参数 ====================')
print(f'数据损失权重: {weight_data}, 解析损失权重: {weight_analytical}')
print(f'最优决策树数量: {best_n_estimators}')
print(f'最优树最大深度: {best_max_depth}')
print(f'训练集最优总MSE: {best_loss:.4f}')

# 用最优参数训练最终模型
final_rf = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth)
final_rf.fit(X_train, y_train.ravel())