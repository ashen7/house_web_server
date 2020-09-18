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

#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
#用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# house_data_file = "app/house_data_collection/data/广州二手房清洗后数据.csv"
house_data_file = "../house_data_collection/data/广州二手房清洗后数据.csv"
sample_scaler = MinMaxScaler(feature_range=(0,1))
label_scaler = MinMaxScaler(feature_range=(0,1))

community_dict = dict()     # 小区名称
shape_dict = dict()         # 房屋户型
floor_dict = dict()         # 所在楼层
orientation_dict = dict()   # 房屋朝向
fix_up_dict = dict()        # 装修情况
lift_dict = dict()          # 配备电梯
type_dict = dict()          # 交易权属
use_dict = dict()           # 房屋用途
years_dict = dict()         # 房屋年限
features_list = list()      # 特征列表
regress_test_loss = list()  # 线性回归测试的loss
forest_test_loss = list()   # 随机森林测试的loss

# 数据预处理
def data_preprocess():
    global community_dict
    global shape_dict
    global floor_dict
    global orientation_dict
    global fix_up_dict
    global lift_dict
    global type_dict
    global use_dict
    global years_dict
    global features_list
    global sample_scaler
    global label_scaler

    miss_value = ["null", "暂无数据"]
    house_data = pd.read_csv(house_data_file, header=0, na_values=miss_value)
    label_data = house_data['总价(万元)']
    labels = np.array(label_data, dtype=np.float32)  # 总价(万元)

    use_col_index_list = [1, 5, 6, 7, 11, 13, 15, 18, 20, 21]
    for index in use_col_index_list:
        features_list.append(house_data.columns[index])

    nouse_col_index_list = list()
    for i in range(house_data.shape[1]):
        if i not in use_col_index_list:
            nouse_col_index_list.append(i)

    house_data.drop(house_data.columns[nouse_col_index_list], axis=1, inplace=True)

    samples = list()  # 小区名称, 房屋户型, 所在楼层, 建筑面积(㎡), 房屋朝向, 装修情况, 配备电梯, 交易权属, 房屋用途, 房屋年限
    count = 0

    for n in range(house_data.shape[1]):
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
                sample_dict = lift_dict
            elif n == 7:
                sample_dict = type_dict
            elif n == 8:
                sample_dict = use_dict
            elif n == 9:
                sample_dict = years_dict

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

    for row in range(house_data.shape[0]):
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
                if house_data.iloc[row, col] == "暂无数据" or math.isnan(float(house_data.iloc[row, col])):
                    house_data.drop(row)
                    labels = np.delete(labels, row)
                    flag = True
                    break
                data.append(float(house_data.iloc[row, col]))
                continue
            elif col == 4:
                sample_dict = orientation_dict
            elif col == 5:
                sample_dict = fix_up_dict
            elif col == 6:
                sample_dict = lift_dict
            elif col == 7:
                sample_dict = type_dict
            elif col == 8:
                sample_dict = use_dict
            elif col == 9:
                sample_dict = years_dict
            data.append(sample_dict[house_data.iloc[row, col]])
        if not flag:
            samples.append(data)
    samples = np.array(samples, dtype=np.float32)
    sample_scaler = sample_scaler.fit(samples)
    samples = sample_scaler.transform(samples)

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
    train_sample_count = int(0.8 * len(samples))
    train_label_count = int(0.8 * len(labels))

    train_sample = samples[0:train_sample_count]
    test_sample = samples[train_sample_count:]
    train_label = labels[0:train_label_count]
    test_label = labels[train_label_count:]

    return train_sample, test_sample, train_label, test_label

# 建立多元线性回归模型
def build_model(input, weights, bias):
    with tf.name_scope("regression_model"):
        return tf.matmul(input, weights) + bias

# 画出训练损失函数的趋势图
def plot_train_loss(train_loss_list):
    plt.title('Train Loss Function')
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.plot(range(0, len(train_loss_list)), train_loss_list, 'g', label='train loss')
    plt.legend(loc='best')
    plt.show()

# 画出测试损失函数的趋势图
def plot_test_loss(test_loss_list):
    plt.title('Test Loss Function')
    plt.xlabel('epoch')
    plt.ylabel('test loss')
    plt.plot(range(0, len(test_loss_list)), test_loss_list, 'r', label='test loss')
    plt.legend(loc='best')
    plt.show()

def plot_loss():
    plt.title('随机森林Test Loss | 线性回归Test Loss')
    plt.xlabel('test number')
    plt.ylabel('test loss')
    plt.plot(range(0, len(forest_test_loss)), forest_test_loss, label='随机森林', color='r')
    plt.plot(range(0, len(regress_test_loss)), regress_test_loss, label='线性回归', color='g')
    plt.legend(loc='best')
    plt.show()

# 训练
def train(train_sample, train_label, test_sample, test_label):
    train_loss_list = list()
    test_loss_list = list()
    epochs = 500
    lr = 0.05
    batch_size = 100

    # 定义tf占位符 输入输出格式
    x = tf.placeholder(dtype=tf.float32, shape=[None, train_sample.shape[1]], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
    # 权重
    weights = tf.Variable(tf.truncated_normal([train_sample.shape[1], 1], dtype=tf.float32,
                                               stddev=1e-1), name="weights")
    bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1]),
                       trainable=True, name="bias")

    # 建立网络
    predict = build_model(x, weights, bias)
    # 定义训练的loss函数 均方误差
    loss = tf.reduce_mean(tf.square(y - predict))
    # 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    # saver持久化 保存模型
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        # 编译 静态图 初始化变量
        session.run(init)
        # 加载模型
        model = tf.train.get_checkpoint_state("app/house_predict/model")
        if model and model.model_checkpoint_path:
            saver.restore(session, model.model_checkpoint_path)
            print('Successfully Load Model!')

        for epoch in tqdm(range(epochs)):
            loss_average = 0
            train_sample, train_label = shuffle(train_sample, train_label)

            batch_sample = list()
            batch_label = list()
            count = 0
            for sample, label in zip(train_sample, train_label):
                batch_sample.append(sample)
                batch_label.append(label)
                count += 1
                # sample = sample.reshape(1, train_sample.shape[1])
                # label = label.reshape(1, 1)
                if count % batch_size == 0:
                    batch_sample_np = np.array(batch_sample)
                    batch_label_np = np.array(batch_label)
                    batch_label_np = batch_label_np.reshape(-1, 1)
                    _, train_loss = session.run([optimizer, loss], feed_dict={
                        x: batch_sample_np,
                        y: batch_label_np
                    })
                    batch_sample.clear()
                    batch_label.clear()
                    loss_average += train_loss

            loss_average /= len(train_sample)
            train_loss_list.append(loss_average)
            w = weights.eval(session=session)
            b = bias.eval(session=session)
            print("Epoch: {}, train loss={}, w={}, b={}".format(epoch + 1, loss_average, w, b))

            # 每训练5 epoch 测试一次 保存模型
            if (epoch + 1) % 5 == 0:
                loss_average = 0
                test_sample, test_label = shuffle(test_sample, test_label)
                for sample, label in zip(test_sample, test_label):
                    sample = sample.reshape(1, test_sample.shape[1])
                    label = label.reshape(1, 1)

                    test_loss = session.run(loss, feed_dict={
                        x: sample,
                        y: label
                    })
                    loss_average += test_loss
                loss_average /= len(test_sample)
                test_loss_list.append(loss_average)
                print("===================Epoch: {}, test loss={}===================".format(epoch + 1, loss_average))

                # 保存模型
                # saver.save(session, "app/house_predict/model/house_predict_model.ckpt")
                saver.save(session, "../house_predict/model/house_predict_model.ckpt")

        # saver.save(session, "app/house_predict/model/house_predict_model.ckpt")
        saver.save(session, "../house_predict/model/house_predict_model.ckpt")
        plot_train_loss(train_loss_list)
        plot_test_loss(test_loss_list)

# 测试
def test(test_sample, test_label):
    global regress_test_loss
    # 定义tf占位符 输入输出格式
    x = tf.placeholder(dtype=tf.float32, shape=[None, test_sample.shape[1]], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
    # 权重
    weights = tf.Variable(tf.truncated_normal([test_sample.shape[1], 1], dtype=tf.float32,
                                              stddev=1e-1), name="weights")
    bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1]),
                       trainable=True, name="bias")

    # 建立网络
    predict = build_model(x, weights, bias)
    # 定义训练的loss函数 均方误差
    loss = tf.reduce_mean(tf.square(y - predict))
    # saver持久化 保存模型
    saver = tf.train.Saver()

    with tf.Session() as session:
        # 加载模型
        # model = tf.train.get_checkpoint_state("app/house_predict/model")
        model = tf.train.get_checkpoint_state("./model")
        if model and model.model_checkpoint_path:
            saver.restore(session, model.model_checkpoint_path)
            print('Successfully Load Linear Regression Model!')

            loss_average = 0
            rmse = 0
            predict_output_list = list()
            for sample, label in zip(test_sample, test_label):
                sample = sample.reshape(1, test_sample.shape[1])
                label = label.reshape(1, 1)

                predict_output, test_loss = session.run([predict, loss], feed_dict={
                    x: sample,
                    y: label
                })
                loss_average += test_loss
                predict_output_list.append(predict_output[0][0])

                #
                # scale = scale_label / label
                # scale_predict_output = scale * predict_output
                # rmse += np.square(scale_label[0][0] - scale_predict_output[0][0])
            print(predict_output_list)
            print(test_label)
            test_label = test_label.reshape(-1, 1)
            test_label = label_scaler.inverse_transform(test_label)
            test_label = test_label.reshape(-1)
            predict_output_list = np.array(predict_output_list).reshape(-1, 1)
            predict_output_list = label_scaler.inverse_transform(predict_output_list)
            predict_output_list = predict_output_list.reshape(-1)
            print(test_label)
            print(predict_output_list)
            rmse = np.sqrt(np.mean(np.square(predict_output_list - test_label)))

            [print("model predict house total price: {}(万元), ground truth house total price: {}(万元)".format(
                    scale_predict_output, scale_label)) for scale_predict_output,scale_label in zip(predict_output_list,test_label)]
            for y_hat, y in zip(predict_output_list, test_label):
                regress_test_loss.append(np.sqrt(abs(y_hat - y)))

            print("Test average loss: ", loss_average / len(test_sample))
            print("RMSE: ", rmse)

# 接口 预测价格
def predict_total_price(sample):
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
            sample_dict = lift_dict
        elif i == 7:
            sample_dict = type_dict
        elif i == 8:
            sample_dict = use_dict
        elif i == 9:
            sample_dict = years_dict
        data.append(sample_dict[sample[i]])
    sample = np.array(data, dtype=np.float32).reshape(1, -1)
    sample = sample_scaler.transform(sample)

    # 定义tf占位符 输入输出格式
    x = tf.placeholder(dtype=tf.float32, shape=[None, sample.shape[1]], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
    # 权重
    weights = tf.Variable(tf.truncated_normal([sample.shape[1], 1], dtype=tf.float32,
                                              stddev=1e-1), name="weights")
    bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1]),
                       trainable=True, name="bias")

    # 建立网络
    predict = build_model(x, weights, bias)
    # saver持久化 保存模型
    saver = tf.train.Saver()

    with tf.Session() as session:
        # 加载模型
        model = tf.train.get_checkpoint_state("app/house_predict/model")
        if model and model.model_checkpoint_path:
            saver.restore(session, model.model_checkpoint_path)
            print('Successfully Load Model!')

            predict_output = session.run(predict, feed_dict={
                x: sample
            })
            predict_output = label_scaler.inverse_transform(predict_output)
            print("model predict house total price: {}(万元)".format(predict_output[0][0]))
    return predict_output[0][0]

# 随机森林训练
def train_random_forest_regression(train_sample, train_label, test_sample, test_label):
    trees = 10     #CART回归树数量
    n_features = train_sample.shape[1] #特征个数
    max_depth = 20  #数的最大深度

    random_forest = RandomForestRegressor(n_estimators=1500, verbose=2, random_state=42, n_jobs=6)
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

    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance');
    plt.xlabel('Variable');
    plt.title('Variable Importances');
    plt.show()

    # 保存模型
    # joblib.dump(random_forest, "app/house_predict/model/random_forest.pkl")
    joblib.dump(random_forest, "./model/random_forest.pkl")

def test_random_forest_regression(test_sample, test_label):
    global forest_test_loss
    # random_forest = joblib.load("app/house_predict/model/random_forest.pkl")
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
    for y_hat, y in zip(predict_result, test_label):
        forest_test_loss.append(np.sqrt(abs(y_hat - y)))

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
            sample_dict = lift_dict
        elif i == 7:
            sample_dict = type_dict
        elif i == 8:
            sample_dict = use_dict
        elif i == 9:
            sample_dict = years_dict
        data.append(sample_dict[sample[i]])
    sample = np.array(data, dtype=np.float32).reshape(1, -1)
    sample = sample_scaler.transform(sample)

    # 加载模型
    random_forest = joblib.load("app/house_predict/model/random_forest.pkl")
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
        # return predict_total_price(source_data)
        predict_result = random_forest_predict_total_price(source_data)
        return round(predict_result, 2)
    except Exception as e:
        print(e)


def main():
    # 数据预处理 并得到训练/测试数据集
    train_sample, test_sample, train_label, test_label = get_datasets()
    # 测试
    source_data = ['金碧御水山庄','4室2厅1厨2卫','高楼层 (共12层)',176.65, '南',
                   '精装','有','商品房','普通住宅','满五年']

    # 随机森林回归
    train_random_forest_regression(train_sample, train_label, test_sample, test_label)
    test_random_forest_regression(test_sample, test_label)
    # random_forest_predict_total_price(source_data)

    # 多元线性回归
    # train(train_sample, train_label, test_sample, test_label)
    test(test_sample, test_label)
    # predict_total_price(source_data)

    plot_loss()

if __name__ == '__main__':
    main()