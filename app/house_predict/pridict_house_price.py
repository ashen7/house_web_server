import os
import math
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

# acc = cross_val_score(model, samples, labels, cv=10, scoring='accuracy', n_jobs=-1).mean()
# test_api = True  #True为视图调用路径，False为内部单独执行路径
test_api = False
if test_api:
    house_data_file = "app/house_data_collection/data/广州二手房清洗后数据.csv"
else:
    house_data_file = "../house_data_collection/data/广州二手房清洗后数据.csv"
sample_scaler = MinMaxScaler(feature_range=(0,1))
label_scaler = MinMaxScaler(feature_range=(0,1)) #通过sklearn中的preprocessing.MinMaxScaler实现归一化

community_dict = dict()     # 小区名称
shape_dict = dict()         # 房屋户型,delete
floor_dict = dict()         # 所在楼层,delete
orientation_dict = dict()   # 房屋朝向
fix_up_dict = dict()        # 装修情况
lift_scale_dict = dict()    # 梯户比例
lift_dict = dict()          # 配备电梯

features_list = list()

# 数据预处理
def data_preprocess():
    global community_dict
    global shape_dict
    global floor_dict
    global orientation_dict
    global fix_up_dict
    global lift_dict
    global lift_scale_dict
    global features_list #特征列表，8个特征
    global sample_scaler
    global label_scaler

    miss_value = ["null", "暂无数据"]#过滤无用数据
    house_data = pd.read_csv(house_data_file, header=0, na_values=miss_value,encoding='utf-8',engine='python')#读取清洗后的数据
    label_data = house_data['总价(万元)']
    labels = np.array(label_data, dtype=np.float32)  # 总价(万元)，存储房价浮点型的数组

    use_col_index_list = [1, 5, 6, 7, 11, 13, 14, 15] #下标指定选择属性，对应清洗后数据
    for index in use_col_index_list:
        features_list.append(house_data.columns[index]) #行、列索引

    nouse_col_index_list = list()
    for i in range(house_data.shape[1]):
        if i not in use_col_index_list:
            nouse_col_index_list.append(i)

    house_data.drop(house_data.columns[nouse_col_index_list], axis=1, inplace=True)#从读入的数据列表中删除掉非指定属性

    samples = list()  # 小区名称, 房屋户型, 所在楼层, 建筑面积(㎡), 房屋朝向, 装修情况, 配备电梯, 交易权属, 房屋用途, 房屋年限
    count = 0

    for n in range(house_data.shape[1]):#把处理好的数据列表内容按照其属性的类别分级，定性变量数值化
        count = 0
        for sample in house_data[house_data.columns[n]]:
            if n == 0:
                sample_dict = community_dict
            elif n == 1:
                sample_dict = shape_dict
            elif n == 2:
                sample_dict = floor_dict
            elif n == 3:
                continue
            elif n == 4:
                sample_dict = orientation_dict
            elif n == 5:
                sample_dict = fix_up_dict
            elif n == 6:
                sample_dict = lift_scale_dict
            elif n == 7:
                sample_dict = lift_dict

            if sample not in sample_dict:
                sample_dict[sample] = count
                count += 1
    # print(community_dict)
    # print(shape_dict)
    # print(floor_dict)
    # print(orientation_dict)
    # print(fix_up_dict)
    # print(lift_dict)
    # print(type_dict)
    # print(use_dict)
    # print(years_dict)

    delete_label_col_list = list()

    for row in range(house_data.shape[0]):#数值化的数据再次清洗并保存到浮点型数组
        data = list()
        flag = False
        for col in range(house_data.shape[1]):
            if col == 0:
                sample_dict = community_dict
            elif col == 1:
                sample_dict = shape_dict
            elif col == 2:
                sample_dict = floor_dict
            elif col == 3:
                if house_data.iloc[row, col] == "暂无数据" or math.isnan(float(house_data.iloc[row, col])) or not labels[row]:
                    house_data.drop(row)#删除掉有"暂无数据"字段的行
                    delete_label_col_list.append(row)
                    flag = True
                    break
                data.append(float(house_data.iloc[row, col]))
                continue
            elif col == 4:
                sample_dict = orientation_dict
            elif col == 5:
                sample_dict = fix_up_dict
            elif col == 6:
                sample_dict = lift_scale_dict
            elif col == 7:
                sample_dict = lift_dict
            data.append(sample_dict[house_data.iloc[row, col]])
        if not flag:
            samples.append(data)
    samples = np.array(samples, dtype=np.float32)
    sample_scaler = sample_scaler.fit(samples)
    samples = sample_scaler.transform(samples)

    delete_count = 0
    for col in delete_label_col_list:
        labels = np.delete(labels, col + delete_count)
        delete_count -= 1

    labels = labels.reshape(-1, 1)
    label_scaler = label_scaler.fit(labels)
    labels = label_scaler.transform(labels)
    labels = labels.reshape(-1)

    # for col in range(samples.shape[1]):
    #     samples[:, col] = (samples[:, col] - samples[:, col].min()) / (samples[:, col].max() - samples[:, col].min())

    # labels = (labels - labels.min()) / (labels.max() - labels.min())

    return samples, labels

# 得到数据集
def get_datasets():
    samples, labels = data_preprocess()
    print(samples.shape)
    print(labels.shape)
    train_sample_count = int(0.8 * len(samples))   #0.8作为训练数据，0.2作为测试数据
    train_label_count = int(0.8 * len(labels))

    train_sample = samples[0:train_sample_count]
    test_sample = samples[train_sample_count:]
    train_label = labels[0:train_label_count]
    test_label = labels[train_label_count:]

    return train_sample, test_sample, train_label, test_label


# 随机森林训练
def train_random_forest_regression(train_sample, train_label, test_sample, test_label):
    trees = 10     #CART回归树数量
    n_features = train_sample.shape[1] #特征个数
    max_depth = 20  #树的最大深度

    random_forest = RandomForestRegressor(n_estimators=1000, verbose=2, random_state=42, n_jobs=6)
    random_forest.fit(train_sample, train_label)
    score = random_forest.score(test_sample, test_label)

    predict = random_forest.predict(test_sample)
    errors = abs(predict - test_label)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    acc = 100 - np.mean((100 * (errors / test_label)))
    print('Test Accuracy: {}%'.format(round(acc, 2)))

    feature_list = list()

    importances = list(random_forest.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x:x[1], reverse=True)
    [print("Variable: {:20} Importance: {}".format(*pair)) for pair in feature_importances];

    x_values = list(range(len(importances))) #画图
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance');
    plt.xlabel('Variable');
    plt.title('Variable Importances');
    plt.show()

    # 保存模型
    if test_api:
        joblib.dump(random_forest, "app/house_predict/model/random_forest.pkl")
    else:
        joblib.dump(random_forest, "./model/random_forest.pkl")

def test_random_forest_regression(test_sample, test_label):  #测试模型
    if test_api:
        random_forest = joblib.load("app/house_predict/model/random_forest.pkl")
    else:
        random_forest = joblib.load("./model/random_forest.pkl")
    print('Successfully Load Random Forest Regression Model!')
    predict_result = random_forest.predict(test_sample)
    rmse = 0

    test_label = test_label.reshape(-1, 1)
    test_label = label_scaler.inverse_transform(test_label)
    test_label = test_label.reshape(-1)
    predict_result = np.array(predict_result).reshape(-1, 1)
    predict_result = label_scaler.inverse_transform(predict_result)
    predict_result = predict_result.reshape(-1)
    print(test_label)
    print(predict_result)
    rmse = np.sqrt(np.mean(np.square(predict_result - test_label)))

    [print("model predict house total price: {}(万元), ground truth house total price: {}(万元)".format(
        scale_predict_output, scale_label)) for scale_predict_output, scale_label in
        zip(predict_result, test_label)]
    print("RMSE: ", rmse)


# 接口 预测价格
def random_forest_predict_total_price(sample):
    data = list()
    for i in range(len(sample)):
        if i == 0:
            sample_dict = community_dict
        elif i == 1:
            sample_dict = shape_dict
        elif i == 2:
            sample_dict = floor_dict
        elif i == 3:
            if sample[i] == "暂无数据" or math.isnan(float(sample[i])):
                return
            data.append(float(sample[i]))
            continue
        elif i == 4:
            sample_dict = orientation_dict
        elif i == 5:
            sample_dict = fix_up_dict
        elif i == 6:
            sample_dict = lift_scale_dict
        elif i == 7:
            sample_dict = lift_dict

        data.append(sample_dict[sample[i]])
    sample = np.array(data, dtype=np.float32).reshape(1, -1)
    sample = sample_scaler.transform(sample)

    # 加载模型
    if test_api:
        random_forest = joblib.load("app/house_predict/model/random_forest.pkl")
    else:
        random_forest = joblib.load("./model/random_forest.pkl")
    print('Successfully Load Random Forest Regression Model!')
    predict_result = random_forest.predict(sample)

    predict_result = np.array(predict_result).reshape(-1, 1)
    predict_result = label_scaler.inverse_transform(predict_result)
    predict_result = predict_result.reshape(-1)

    print("model predict house total price: {}(万元)".format(predict_result[0]))
    return predict_result[0]

def predict_house_price_api(source_data):
    data_preprocess()
    try:
        predict_result = random_forest_predict_total_price(source_data)
        return round(predict_result, 2)
    except Exception as e:
        print(e)

def main():
    # 数据预处理 并得到训练/测试数据集
    train_sample, test_sample, train_label, test_label = get_datasets()
    # 测试
    source_data = ['金碧御水山庄', '4室2厅1厨2卫', '高楼层 (共12层)', 176.65, '南',
                   '精装', '一梯两户','有']
    # 随机森林回归
    train_random_forest_regression(train_sample, train_label, test_sample, test_label)
    test_random_forest_regression(test_sample, test_label)
    random_forest_predict_total_price(source_data)

if __name__ == '__main__':
    main()