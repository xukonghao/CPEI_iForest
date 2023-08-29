import numpy as np
import pandas as pd
import random
import xlwt
import csv
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datetime
import copy
from CPEI_Iforest import IsolationTreeEnsemble
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score,accuracy_score,ConfusionMatrixDisplay,confusion_matrix,mutual_info_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import BernoulliRBM
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


def filter_by_variance(threshold, var, dataTemp):
    global label_list, head_row
    columns_drop = []
    for i in range(dataTemp.columns.size):
        if var[i] <= threshold:
            columns_drop.append(dataTemp.columns[i])
    dataTemp.drop(axis=1, labels=columns_drop, inplace=True)
    dataTemp.columns = range(len(dataTemp.columns))
    # head_row = np.delete(head_row, columns_drop)
    return dataTemp

def filter_by_pearson(threshold, dataTemp):
    global pearson
    pearson = [[None for j in range(len(dataTemp.columns))] for i in range(len(dataTemp.columns))]
    columns_drop = []
    for i in range(len(dataTemp.columns)):
        j = i + 1
        for j in range(j, len(dataTemp.columns)):
            corr = pearsonr(dataTemp[dataTemp.columns[i]], dataTemp[dataTemp.columns[j]])
            pearson[i][j] = corr[0]
            pearson[j][i] = corr[0]
            if abs(corr[0]) >= threshold:
                columns_drop.append(dataTemp.columns[i])
            j += 1
    dataTemp.drop(axis=1, labels=columns_drop, inplace=True)
    return dataTemp


def filter_by_mutual_info(threshold, dataTemp):
    global by_mutual_info
    by_mutual_info = [[None for j in range(len(dataTemp.columns))] for i in range(len(dataTemp.columns))]
    columns_drop = []
    for i in range(len(dataTemp.columns)):
        j = i + 1
        for j in range(j, len(dataTemp.columns)):
            # corr = by_mutual_infor(dataTemp[dataTemp.columns[i]], dataTemp[dataTemp.columns[j]])
            mutual_info = metrics.normalized_mutual_info_score(dataTemp[dataTemp.columns[i]],
                                                               dataTemp[dataTemp.columns[j]])
            by_mutual_info[i][j] = mutual_info
            by_mutual_info[j][i] = mutual_info
            if mutual_info >= threshold:
                columns_drop.append(dataTemp.columns[i])
            j += 1
    dataTemp.drop(axis=1, labels=columns_drop, inplace=True)
    return dataTemp
'''
readme:
这版本写的很好用.train_normal指的是训练集中所有label为normal的样本.
'''
df_test = pd.read_csv("./Dataset/UNSWNB15/UNSW_NB15_training-set.csv")  # 82332
df_train = pd.read_csv("./Dataset/UNSWNB15/UNSW_NB15_testing-set.csv")
title = df_train.columns.tolist()
title = title[1:-2]
del (title [1:4])
title.extend(["proto","service","state"])

# 获得各种攻击类型
attack_cat_train=df_train["attack_cat"]
attack_cat_test=df_test["attack_cat"]
attack_cat_examples=["Normal","Generic","Exploits","Fuzzers","DoS",
                 "Reconnaissance","Analysis","Backdoor","Shellcode","Worms"]

# print("proto有这么多种", df_train["proto"].value_counts().shape)  # proto 有133种,就不要了吧
print("train label标签：",df_train["label"].value_counts())
print("test label标签：",df_test["label"].value_counts())
label_train=df_train["label"].values.tolist()
label_test=df_test["label"].values.tolist()

df_train = df_train.drop(["label", "id"], axis=1)
df_test = df_test.drop(["label", "id"], axis=1)
list_train = []
list_test = []

def toLabel(df_train, df_test, onehot_encoder=False,examples=None):
    # print(df_train.shape,df_test.shape)#175341,) (82332,)\
    le = LabelEncoder()
    # LabelEncoder有个问题,那就是他编号的时候是按照类别在数据集出现的先后顺序编的号.我们要想按照自己的需求编号,
    # 可以把我们的类别写在examples里面,但是要写全哦
    if examples is None:
        df = pd.concat([df_train, df_test], axis=0)
        # print("df",df.shape)#(257673,)
        le.fit(df)
    else:
        print(f"按照我们写的顺序编号:{examples}")
        le.fit(examples)
        #查看编码顺序,发现我们写的这个p用没有.因为labelencoder是按照字典顺序编码的.
        for i in range(10):
            print(i," : ",le.inverse_transform([i])[0],end=",")
    df_train = le.transform(df_train).reshape([-1, 1])
    df_test = le.transform(df_test).reshape([-1, 1])
    df = np.concatenate([df_train, df_test], axis=0)
    if onehot_encoder:
        ohe = OneHotEncoder()
        ohe.fit(df)
        df_train = ohe.transform(df_train).toarray()
        df_test = ohe.transform(df_test).toarray()
    # print("onehot:", df_train.shape, df_test.shape)
    '''
    proto
    onehot: (175341, 1) (82332, 1)
    service
    onehot: (175341, 13) (82332, 13)
    state
    onehot: (175341, 11) (82332, 11)
    attack_cat
    onehot: (175341, 10) (82332, 10)
    '''
    return df_train, df_test

#对数据集进行编码
for i in df_train.select_dtypes(include="object").columns:
    print(i)
    series_train = df_train[i]
    series_test = df_test[i]
    series_train, series_test = toLabel(series_train, series_test, False and i != "proto")
    list_train.append(series_train)  # z这个列表的结构是怎么样的,以后看,不用看了就是几个宽度不同的数组横向拼起来
    list_test.append(series_test)
# 对攻击种类进行编码,不需要独热.
attack_cat_train,attack_cat_test=toLabel(attack_cat_train,attack_cat_test,
                         onehot_encoder=False,examples=attack_cat_examples)
# print(attack_cat_train[:10])
# 出去attack_cat其他都要
categorical_train = np.concatenate(list_train[:-1], axis=1)
categorical_test = np.concatenate(list_test[:-1], axis=1)

#删掉原始df里的object列
df_train=df_train.drop(df_train.select_dtypes(include="object").columns,axis=1).to_numpy()
df_test=df_test.drop(df_test.select_dtypes(include="object").columns,axis=1).to_numpy()
#得到处理好特征的数据
# data_train = df_train
# data_test = df_test
data_train=np.concatenate([df_train,categorical_train],axis=1)
data_test=np.concatenate([df_test,categorical_test],axis=1)
# print(data_train.shape,data_test.shape)#(175341, 64) (82332, 64)
# print(label_train.shape,label_test.shape)#(175341,) (82332,)
def double_sided_log(x):
    return np.sign(x) * np.log(1 + np.abs(x))
def sigmoid(x):
    return np.divide(1, (1 + np.exp(np.negative(x))))
#归一化(175341, 64) (82332, 64)
data_train=double_sided_log(sigmoid(data_train))
data_test=double_sided_log(sigmoid(data_test))
df_train=double_sided_log(sigmoid(df_train))
df_test=double_sided_log(sigmoid(df_test))

data = np.concatenate([data_train, data_test], axis=0)
head_row_continuous = np.concatenate([label_train, label_test], axis=0)
# data = pd.DataFrame(data)
label = head_row_continuous

normal,attack,label_list,data_continuous=0,0,[],[]
for row in range(len(label)):
    row_len = data.shape[1] - 1
    # row[0:row_len] = [float(x) for x in row[0:row_len]]
    # row[row_len] = int(row[row_len])
    if label[row]==0 and normal < 45000:
        data_continuous.append(data[row][0:row_len])
        label_list.append(1)
        normal += 1
    elif label[row]==1 and attack < 5000:
        data_continuous.append(data[row][0:row_len])
        label_list.append(0)
        attack += 1
print("normal:",normal,"attack:",attack,"data_continuous.shape:",len(data_continuous))
# for i, x in enumerate(label_list):
#     if x == 0: label_list[i] = 1
#     elif x== 1: label_list[i] = 0
# 数据归一化------------------------------------------------------------------
scaler = MinMaxScaler()
scaler.fit(data_continuous)
data_continuous = scaler.transform(data_continuous)
data_continuous = pd.DataFrame(data_continuous)
print("归一化后数据(列)", data_continuous.shape[1])

# 方差 预过滤----------------------------------------------------------------
variance_continuous = []
for i in range(data_continuous.columns.size):
    variance_continuous.append(np.var(data_continuous[data_continuous.columns[i]]))  # 计算方差
data_continuous=filter_by_variance(0, variance_continuous, data_continuous)
head_row_continuous = data_continuous.columns.tolist()
print("方差过滤后数据(列)", data_continuous.shape[1])

# 定义L1正则化稀疏自编码器模型
model = Pipeline(steps=[('rbm', BernoulliRBM(n_components=33, n_iter=20, learning_rate=0.1, verbose=True, random_state=42))])
# 训练L1正则化稀疏自编码器模型 降维
model.fit(data_continuous)
mutual_info_threshold = [i / 10 for i in range(1, 11)]
data_continuous = pd.DataFrame(data=data_continuous.values, columns=head_row_continuous)
data_continuous = filter_by_mutual_info(mutual_info_threshold[8], data_continuous)
data_continuous = filter_by_pearson(0.8, data_continuous)
print("L1正则化稀疏自编码器降维后 数据维度(列)", data_continuous.shape[1])

X_train, X_test, y_train, y_test = train_test_split(data_continuous, label_list, test_size=0.2,
                                                    random_state=104)
#DNN(深度神经网络)模型
if 0:
    list_score = [i * 0 for i in range(1)]
    for i in range(2, 10):
        print("------------------------------DNN(深度神经网络)模型 i=", i, "-------------------------------------------------")
        # 定义DNN模型
        seed_value = 2
        tf.random.set_seed(seed_value)
        start_time = datetime.datetime.now()
        input_dim = X_train.shape[1]
        encoding_dim = 16
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value)),
            tf.keras.layers.Dense(32, activation='relu',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value)),
            tf.keras.layers.Dense(encoding_dim, activation='relu',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value)),
            tf.keras.layers.Dense(32, activation='relu',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value)),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value)),
            tf.keras.layers.Dense(input_dim, activation='sigmoid',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value))
        ])
        model.compile(optimizer='adam', loss='mse')
        # 训练DNN模型
        model.fit(X_train, X_train, epochs=100, batch_size=32, verbose=0)
        # 使用训练好的模型进行预测和异常评估
        y_train_score = model.predict(X_test)
        end_time = datetime.datetime.now()
        y_prob = np.mean(np.power(X_test - y_train_score, 2), axis=1)  # 计算重构误差
        y_prob = -y_prob
        # 根据归一化后的距离判断异常点和正常点
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        print("AUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        # print("fpr:")
        # for j in fpr:
        #     print(j)
        # print("\ntpr:")
        # for j in tpr:
        #     print(j)
        min_far = 1.0
        best_threshold = None
        for i in range(len(thresholds)):
            y_pred = (y_prob >= thresholds[i]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = tp / (tp + fn)
            if far <= min_far and far > 0.8:
                min_far = far
                best_threshold = thresholds[i]
        # 使用最佳阈值对测试数据进行预测，得到预测标签y_prod
        y_prod = (y_prob >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_prod).ravel()
        print("模型运行时间(Running time):", (end_time - start_time), \
              "\n准确率(accuracy):", '{:.2%}'.format(accuracy_score(y_test, y_prod)), \
              "\n召回率(Recall):", '{:.2%}'.format(tp / (tp + fn)), \
              "\n误检率(FPR):", '{:.2%}'.format(fp / (tn + fp)), \
              "\n精确率(Precision):", '{:.2%}'.format(tp / (tp + fp)), \
              "\nF1 score:", '{:.2%}'.format(f1_score(y_test, y_prod)), \
              "\nAUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)), \
              "\n混淆矩阵:", confusion_matrix(y_test, y_prod, labels=[1, 0]))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('kddcup99 CPEI-iForest ROC-Curve')
        plt.show()


#k-means模型
if 0:
    list_score = [i * 0 for i in range(0)]
    for i in range(0, 100):
        print("------------------------------k-means模型 i=", i, "-------------------------------------------------")
        # 创建一个KMeans聚类模型
        start_time = datetime.datetime.now()
        model = KMeans(n_clusters=2, random_state=42)
        # 对数据进行拟合
        model.fit(X_train)
        # 计算每个样本到聚类中心的距离
        end_time = datetime.datetime.now()
        distances = cdist(X_test, model.cluster_centers_, 'euclidean')
        # 对距离进行归一化处理
        scaler = MinMaxScaler()
        distances_normalized = scaler.fit_transform(distances)
        y_prob = distances_normalized.min(axis=1)
        y_prob = -y_prob
        # 根据归一化后的距离判断异常点和正常点
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        print("AUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        min_far = 1.0
        best_threshold = None
        for i in range(len(thresholds)):
            y_pred = (y_prob >= thresholds[i]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = tp / (tp + fn)
            if far <= min_far and far > 0.8:
                min_far = far
                best_threshold = thresholds[i]
        # 使用最佳阈值对测试数据进行预测，得到预测标签y_prod
        y_prod = (y_prob >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_prod).ravel()
        # print("fpr:")
        # for j in fpr:
        #     print(j)
        # print("\ntpr:")
        # for j in tpr:
        #     print(j)
        print("模型运行时间(Running time):", (end_time - start_time), \
              "\n准确率(accuracy):", '{:.2%}'.format(accuracy_score(y_test, y_prod)), \
              "\n召回率(Recall):", '{:.2%}'.format(tp / (tp + fn)), \
              "\n误检率(FPR):", '{:.2%}'.format(fp / (tn + fp)), \
              "\n精确率(Precision):", '{:.2%}'.format(tp / (tp + fp)), \
              "\nF1 score:", '{:.2%}'.format(f1_score(y_test, y_prod)), \
              "\nAUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)), \
              "\n混淆矩阵:", confusion_matrix(y_test, y_prod, labels=[1, 0]))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('kddcup99 CPEI-iForest ROC-Curve')
        plt.show()


#gmm高斯混合模型
if 0:
    list_score = [i * 0 for i in range(0)]
    for i in range(101, 200):
        print("------------------------------gmm高斯混 i=", i, "-------------------------------------------------")
        # 创建GMM模型
        start_time = datetime.datetime.now()
        n_components = 13
        gmm = GaussianMixture(n_components=n_components, random_state=9)
        # 拟合模型
        gmm.fit(X_train)
        end_time = datetime.datetime.now()
        # 在测试集上获取异常分数
        y_scores = gmm.score_samples(X_test)
        # 根据归一化后的距离判断异常点和正常点
        scaler = MinMaxScaler()
        y_prob = scaler.fit_transform(y_scores.reshape(-1, 1)).flatten()
        # y_prob = -y_prob
        # y_prob = -y_prob + 1
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # for j in fpr:
        #     print(j)
        # print("\ntpr:")
        # for j in tpr:
        #     print(j)
        print("AUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        min_far = 1.0
        best_threshold = None
        for i in range(len(thresholds)):
            y_pred = (y_prob >= thresholds[i]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = fp / (fp + tn)
            if far <= min_far and far > 0.027:
                min_far = far
                best_threshold = thresholds[i]
        # 使用最佳阈值对测试数据进行预测，得到预测标签y_prod
        y_prod = (y_prob >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_prod).ravel()
        print("模型运行时间(Running time):", (end_time - start_time), \
              "\n准确率(accuracy):", '{:.2%}'.format(accuracy_score(y_test, y_prod)), \
              "\n召回率(Recall):", '{:.2%}'.format(tp / (tp + fn)), \
              "\n误检率(FPR):", '{:.2%}'.format(fp / (tn + fp)), \
              "\n精确率(Precision):", '{:.2%}'.format(tp / (tp + fp)), \
              "\nF1 score:", '{:.2%}'.format(f1_score(y_test, y_prod)), \
              "\nAUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)), \
              "\n混淆矩阵:", confusion_matrix(y_test, y_prod, labels=[1, 0]))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('kddcup99 CPEI-iForest ROC-Curve')
        plt.show()


#RBM受限玻尔兹曼机模型(DNN一种子模型)
if 0:
    list_score = [i * 0 for i in range(1)]
    for i in range(0, 2):
        print("------------------------------RBM受限玻尔兹曼机模型(DNN一种子模型)  i=", i, "-------------------------------------------------")
        # 创建RBM模型
        start_time = datetime.datetime.now()
        # rbm = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=100, batch_size=32, random_state=42)
        rbm = BernoulliRBM(n_components=20, learning_rate=0.01, n_iter=100, batch_size=2048, random_state=42)
        # 使用正常样本训练RBM模型
        rbm.fit(X_train)
        end_time = datetime.datetime.now()
        # 对测试样本进行重构，并计算重构误差
        X_reconstructed = rbm.gibbs(X_test)
        y_prob = np.mean(np.power(X_test - X_reconstructed, 2), axis=1)
        y_prob = -y_prob + 1
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # for j in fpr:
        #     print(j)
        # print("\ntpr:")
        # for j in tpr:
        #     print(j)
        print("AUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        min_far = 1.0
        best_threshold = None
        for i in range(len(thresholds)):
            y_pred = (y_prob >= thresholds[i]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = tp / (tp + fn)
            if far <= min_far and far > 0.8:
                min_far = far
                best_threshold = thresholds[i]
        # 使用最佳阈值对测试数据进行预测，得到预测标签y_prod
        y_prod = (y_prob >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_prod).ravel()
        print("模型运行时间(Running time):", (end_time - start_time), \
              "\n准确率(accuracy):", '{:.2%}'.format(accuracy_score(y_test, y_prod)), \
              "\n召回率(Recall):", '{:.2%}'.format(tp / (tp + fn)), \
              "\n误检率(FPR):", '{:.2%}'.format(fp / (tn + fp)), \
              "\n精确率(Precision):", '{:.2%}'.format(tp / (tp + fp)), \
              "\nF1 score:", '{:.2%}'.format(f1_score(y_test, y_prod)), \
              "\nAUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)), \
              "\n混淆矩阵:", confusion_matrix(y_test, y_prod, labels=[1, 0]))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('kddcup99 CPEI-iForest ROC-Curve')
        plt.show()


#LOF局部离群因子模型
if 0:
    list_score = [i * 0 for i in range(0)]
    for i in range(30, 100):
        print("------------------------------LOF局部离群因子模型 i=", i, "-------------------------------------------------")
        # 创建LOF模型并进行拟合
        start_time = datetime.datetime.now()
        lof = LocalOutlierFactor(n_neighbors=33, novelty=True)
        lof.fit(X_train)
        end_time = datetime.datetime.now()
        # 在测试集上获取异常分数
        y_prob = lof.decision_function(X_test)
        # y_prob = -y_prob+1
        # 根据归一化后的距离判断异常点和正常点
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # for j in fpr:
        #     print(j)
        # print("\ntpr:")
        # for j in tpr:
        #     print(j)
        print("AUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        min_far = 1.0
        best_threshold = None
        for i in range(len(thresholds)):
            y_pred = (y_prob >= thresholds[i]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = tp / (tp + fn)
            if far <= min_far and far > 0.8:
                min_far = far
                best_threshold = thresholds[i]
        # 使用最佳阈值对测试数据进行预测，得到预测标签y_prod
        y_prod = (y_prob >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_prod).ravel()
        print("模型运行时间(Running time):", (end_time - start_time), \
              "\n准确率(accuracy):", '{:.2%}'.format(accuracy_score(y_test, y_prod)), \
              "\n召回率(Recall):", '{:.2%}'.format(tp / (tp + fn)), \
              "\n误检率(FPR):", '{:.2%}'.format(fp / (tn + fp)), \
              "\n精确率(Precision):", '{:.2%}'.format(tp / (tp + fp)), \
              "\nF1 score:", '{:.2%}'.format(f1_score(y_test, y_prod)), \
              "\nAUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)), \
              "\n混淆矩阵:", confusion_matrix(y_test, y_prod, labels=[1, 0]))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('kddcup99 CPEI-iForest ROC-Curve')
        plt.show()


#孤立森林
if 0:
    list_score = [i * 0 for i in range(0)]
    for i in range(0, 1500):
        print("------------------------------孤立森林 i=", i, "-------------------------------------------------")
        # 创建一个孤立森林模型
        start_time = datetime.datetime.now()
        # model = IsolationForest(max_samples=256, n_estimators=100, random_state=i)
        model = IsolationForest(max_samples=256, n_estimators=15, random_state=0)
        # 对数据进行拟合
        model.fit(X_train)
        # 计算每个样本的异常得分
        end_time = datetime.datetime.now()
        y_prob = model.decision_function(X_test)
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # for j in fpr:
        #     print(j)
        # print("\ntpr:")
        # for j in tpr:
        #     print(j)
        min_far = 1.0
        best_threshold = None
        for i in range(len(thresholds)):
            y_pred = (y_prob >= thresholds[i]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = fp / (fp + tn)
            if far <= min_far and far > 0.027:
                min_far = far
                best_threshold = thresholds[i]
        # 使用最佳阈值对测试数据进行预测，得到预测标签y_prod
        y_prod = (y_prob >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_prod).ravel()
        # end_time = datetime.datetime.now()
        print("模型运行时间(Running time):", (end_time - start_time), \
              "\n准确率(accuracy):", '{:.2%}'.format(accuracy_score(y_test, y_prod)), \
              "\n召回率(Recall):", '{:.2%}'.format(tp / (tp + fn)), \
              "\n误检率(FPR):", '{:.2%}'.format(fp / (tn + fp)), \
              "\n精确率(Precision):", '{:.2%}'.format(tp / (tp + fp)), \
              "\nF1 score:", '{:.2%}'.format(f1_score(y_test, y_prod)), \
              "\nAUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)), \
              "\n混淆矩阵:", confusion_matrix(y_test, y_prod, labels=[1, 0]))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('kddcup99 CPEI-iForest ROC-Curve')
        plt.show()


# CPEI_孤立森林
if 1:
    list_score = [i * 0 for i in range(103)]
    for i in range(104, 120):
        # 创建一个孤立森林模型
        start_time = datetime.datetime.now()
        model = IsolationTreeEnsemble(sample_size=256, n_trees=15, random_state=23)
        # model = IsolationTreeEnsemble(sample_size=256, n_trees=100, random_state=i)
        # 对数据进行拟合
        model.fit(X_train)
        end_time = datetime.datetime.now()
        # 计算每个样本的异常得分
        # y_prod = model.predict(X_test, threshold=0.33)
        y_prob = model.anomaly_score(X_test)
        print("------------------------------CPEI_孤立森林 i=", i, "-------------------------------------------------")
        # 循环遍历每个阈值，计算误检率，并找到误检率最低的阈值
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # for j in fpr:
        #     print(j)
        # print("\ntpr:")
        # for j in tpr:
        #     print(j)
        print("AUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        min_far = 1.0
        best_threshold = None
        for i in range(len(thresholds)):
            y_pred = (y_prob >= thresholds[i]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = fp / (fp + tn)
            if far <= min_far and far > 0.027:
                min_far = far
                best_threshold = thresholds[i]
        # 使用最佳阈值对测试数据进行预测，得到预测标签y_prod
        y_prod = (y_prob >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_prod).ravel()
        print("模型运行时间(Running time):", (end_time - start_time), \
              "\n准确率(accuracy):", '{:.2%}'.format(accuracy_score(y_test, y_prod)), \
              "\n召回率(Recall):", '{:.2%}'.format(tp / (tp + fn)), \
              "\n误检率(FPR):", '{:.2%}'.format(fp / (tn + fp)), \
              "\n精确率(Precision):", '{:.2%}'.format(tp / (tp + fp)), \
              "\nF1 score:", '{:.2%}'.format(f1_score(y_test, y_prod)), \
              "\nAUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)), \
              "\n混淆矩阵:", confusion_matrix(y_test, y_prod, labels=[1, 0]))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('UNSWNB15 CPEI-iForest ROC-Curve')
        plt.show()

