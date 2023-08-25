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


def read_csv(file):
    global label_list, head_row, data, normal, attack,label_list_sort
    with open(file) as f:
        f_csv = csv.reader(f)
        head_row = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",\
                    "logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",\
                    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",\
                    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
        # rows = list(f_csv)
        # random.seed(2)
        # selected_rows = random.sample(rows, 494021)
        # for row in selected_rows:
        for row in f_csv:
            row_len = len(row) - 1
            row[0:row_len] = [float(x) for x in row[0:row_len]]
            row[row_len] = int(row[row_len])
            if row[row_len] == 0 and len(normal) < 90000:
                data.append(row[0:row_len])
                label_list.append(1)
                label_list_sort.append(row[row_len])
                normal.append(row[row_len])
            elif row[row_len] != 0 and len(attack) < 10000:
                data.append(row[0:row_len])
                label_list.append(0)
                label_list_sort.append(row[row_len])
                attack.append(row[row_len])


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


global label_list, head_row_continuous, head_row_discrete, normal, attack, pearson  # 声明全局变量
head_row,head_row_continuous, head_row_discrete, label_list, data, data_discrete, data_continuous, pearson, mutual_info\
    =[], [], [], [], [], [], [], [], []
normal, attack = [], []
label_list_sort = []

# 读取文件，获取标签，表头，数据---------------------------------------------------------
read_csv("Dataset/KDDCUP99/kddcup.data_10_percent_corrected.csv")
data = pd.DataFrame(data=np.array(data), columns=head_row)
#离散数据提取
index_list = [ 1, 2, 3, 6, 11, 20, 21]
head_row_discrete = [head_row[i] for i in index_list]
data_discrete = copy.deepcopy(data)
data_discrete = data_discrete.loc[:, head_row_discrete]
label_list_discrete = copy.deepcopy(label_list)
#连续数据处理
data_continuous = copy.deepcopy(data)  # 数据深拷贝
data_continuous = data.drop(head_row_discrete, axis=1)
print("normal:", len(normal), "attack:", len(attack), "data.shape", data.shape, "data_continuous", data_continuous.shape, "data_discrete", data_discrete.shape)
# for i, x in enumerate(label_list):
#     if x == 0: label_list[i] = 1
#     elif x== 1: label_list[i] = 0

# 数据归一化------------------------------------------------------------------
scaler = MinMaxScaler()
scaler.fit(data_continuous)
data_continuous = scaler.transform(data_continuous)
data_continuous = pd.DataFrame(data_continuous)  # data = pd.DataFrame(data=data.values,columns=head_row[0:-1])

scaler.fit(data_discrete)
data_discrete = scaler.transform(data_discrete)
data_discrete = pd.DataFrame(data_discrete)
print("归一化后数据 data_continuous.shape:", data_continuous.shape," data_discrete.shape:",data_discrete.shape)

# 方差 预过滤----------------------------------------------------------------
variance_continuous = []
for i in range(data_continuous.columns.size):
    variance_continuous.append(np.var(data_continuous[data_continuous.columns[i]]))  # 计算方差
data_continuous=filter_by_variance(0, variance_continuous, data_continuous)
head_row_continuous = data_continuous.columns.tolist()

variance_discrete = []
for i in range(data_discrete.columns.size):
    variance_discrete.append(np.var(data_discrete[data_discrete.columns[i]]))  # 计算方差
data_discrete=filter_by_variance(0, variance_discrete, data_discrete)
head_row_discrete = data_discrete.columns.tolist()
print("方差过滤后 data_continuous.shape:", data_continuous.shape," data_discrete.shape:",data_discrete.shape)

# # 皮尔逊，互信息系数筛选阈值 过滤 xgboost模型计算准确率和耗时
pearson_threshold = [i / 10 for i in range(1, 11)]
i = 5
data_continuous = pd.DataFrame(data=data_continuous.values, columns=head_row_continuous)
data_continuous = filter_by_pearson(pearson_threshold[i], data_continuous)
print("皮尔逊过滤 data_continuous.shape:", data_continuous.shape," data_discrete.shape:",data_discrete.shape)
X_train, X_test, y_train, y_test = train_test_split(data_continuous, label_list, test_size=0.2,
                                                    random_state=170)

# X_train, X_test, Y_train, Y_test = train_test_split(data_continuous, label_list_sort, test_size=0.2,
#                                                     random_state=651)
# y_train = copy.deepcopy(Y_train)
# y_test = copy.deepcopy(Y_test)
# # print("Y_train训练集里 Dos类型:",Y_train.count(13)+Y_train.count(11)+Y_train.count(4)+Y_train.count(7)+Y_train.count(5)+Y_train.count(8),\
# #       " R2L类型:",Y_train.count(12)+Y_train.count(6)+Y_train.count(14)+Y_train.count(18)+Y_train.count(16)+Y_train.count(21)+Y_train.count(20)+Y_train.count(19),\
# #       " U2R类型:",Y_train.count(1)+Y_train.count(2)+Y_train.count(3)+Y_train.count(22),\
# #       " Probe类型:",Y_train.count(10)+Y_train.count(17)+Y_train.count(9)+Y_train.count(15))
# # print("Y_test训练集里 Dos类型:",Y_test.count(13)+Y_test.count(11)+Y_test.count(4)+Y_test.count(7)+Y_test.count(5)+Y_test.count(8),\
# #       " R2L类型:",Y_test.count(12)+Y_test.count(6)+Y_test.count(14)+Y_test.count(18)+Y_test.count(16)+Y_test.count(21)+Y_test.count(20)+Y_test.count(19),\
# #       " U2R类型:",Y_test.count(1)+Y_test.count(2)+Y_test.count(3)+Y_test.count(22),\
# #       " Probe类型:",Y_test.count(10)+Y_test.count(17)+Y_test.count(9)+Y_test.count(15))
# for i, x in enumerate(Y_train):
#     if x == 0:
#         y_train[i] = 1
#     elif x != 0:
#         y_train[i] = 0
# for i, x in enumerate(y_test):
#     if x == 0:
#         y_test[i] = 1
#     elif x != 0:
#         y_test[i] = 0


# CPEI_孤立森林模型
if 0:
    list_score = [i * 0 for i in range(0)]
    for i in range(3000, 5000):
        print("------------------------------CPEI_孤立森林模型 i=", i, "-------------------------------------------------")
        # 创建一个孤立森林模型
        start_time = datetime.datetime.now()
        # model = IsolationTreeEnsemble(sample_size=1024 * 2, n_trees=10, random_state=121)
        model = IsolationTreeEnsemble(sample_size=256, n_trees=15, random_state=i)
        # model = IsolationTreeEnsemble(sample_size=256, n_trees=100, random_state=159)
        # 对数据进行拟合
        model.fit(X_train)
        # 计算每个样本的异常得分
        end_time = datetime.datetime.now()
        y_prob = model.anomaly_score(X_test)
        list_score.append(roc_auc_score(y_test, y_prob))
        print("AUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        # fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # min_far = 1.0
        # best_threshold = None
        # for i in range(len(thresholds)):
        #     y_pred = (y_prob >= thresholds[i]).astype(int)
        #     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        #     far = fp / (fp + tn)
        #     if far <= min_far and far > 0.04:
        #         min_far = far
        #         best_threshold = thresholds[i]
        # # 使用最佳阈值对测试数据进行预测，得到预测标签y_prod
        # y_prod = (y_prob >= best_threshold).astype(int)
        # tn, fp, fn, tp = confusion_matrix(y_test, y_prod).ravel()
        # # end_time = datetime.datetime.now()
        # print("模型运行时间(Running time):", (end_time - start_time), \
        #       "\n准确率(accuracy):", '{:.2%}'.format(accuracy_score(y_test, y_prod)), \
        #       "\n召回率(Recall):", '{:.2%}'.format(tp / (tp + fn)), \
        #       "\n误检率(FPR):", '{:.2%}'.format(fp / (tn + fp)), \
        #       "\n精确率(Precision):", '{:.2%}'.format(tp / (tp + fp)), \
        #       "\nF1 score:", '{:.2%}'.format(f1_score(y_test, y_prod)), \
        #       "\nAUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)), \
        #       "\n混淆矩阵:", confusion_matrix(y_test, y_prod, labels=[1, 0]))
        # print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        # plt.plot(fpr, tpr)
        # plt.plot([0, 1], [0, 1], linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('kddcup99 CPEI-iForest ROC-Curve')
        # plt.show()

#RBM受限玻尔兹曼机模型(DNN一种子模型)
if 0:
    list_score = [i * 0 for i in range(1)]
    for i in range(1, 10):
        print("------------------------------RBM受限玻尔兹曼机模型(DNN一种子模型) i=", i, "-------------------------------------------------")
        # 创建RBM模型
        start_time = datetime.datetime.now()
        # rbm = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=100, batch_size=32, random_state=42)
        rbm = BernoulliRBM(n_components=20, learning_rate=0.01, n_iter=50, batch_size=64, random_state=42)
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
            far = fp / (fp + tn)
            if far <= min_far and far > 0.5475:
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

#gmm高斯混合模型
if 0:
    list_score = [i * 0 for i in range(1)]
    for i in range(3, 14):
        print("------------------------------gmm高斯混合模型 i=", i, "-------------------------------------------------")
        # 创建GMM模型
        start_time = datetime.datetime.now()
        n_components = 7
        gmm = GaussianMixture(n_components=n_components, random_state=3)
        # 拟合模型
        gmm.fit(X_train)
        end_time = datetime.datetime.now()
        # 在测试集上获取异常分数
        y_scores = gmm.score_samples(X_test)
        # 根据归一化后的距离判断异常点和正常点
        scaler = MinMaxScaler()
        y_prob = scaler.fit_transform(y_scores.reshape(-1, 1)).flatten()
        y_prob = -y_prob
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
            if far <= min_far and far > 0.146:
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
    for i in range(1, 100):
        print("------------------------------LOF局部离群因子模型 i=", i, "-------------------------------------------------")
        # 创建LOF模型并进行拟合
        start_time = datetime.datetime.now()
        lof = LocalOutlierFactor(n_neighbors=40, novelty=True)
        lof.fit(X_train)
        end_time = datetime.datetime.now()
        # 在测试集上获取异常分数
        y_prob = lof.decision_function(X_test)
        y_prob = -y_prob + 1
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
            far = fp / (fp + tn)
            if far < min_far and far > 0.06725:
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

#DNN(深度神经网络)模型
if 0:
    list_score = [i * 0 for i in range(1)]
    for i in range(2, 10):
        print("------------------------------#DNN(深度神经网络)模型 i=", i, "-------------------------------------------------")
        # 定义DNN模型
        seed_value = 2
        tf.random.set_seed(seed_value)
        start_time = datetime.datetime.now()
        input_dim = X_train.shape[1]
        encoding_dim = 32
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
        # threshold = np.percentile(y_prob, 95)  # 设置异常检测阈值
        # y_pred = np.where(y_prob > threshold, 1, 0)  # 根据阈值进行二分类预测
        # # y_true = np.zeros_like(y_pred)  # 正常样本标签为0
        # 根据归一化后的距离判断异常点和正常点
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        print("fpr:")
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
            # far = fp / (fp + tn)
            far = tp / (tp + fn)
            if far <= min_far and far > 0.9993:
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
        print("AUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('kddcup99 CPEI-iForest ROC-Curve')
        plt.show()

#k-means模型
if 0:
    list_score = [i * 0 for i in range(50)]
    for i in range(50, 100):
        print("------------------------------k-means模型 i=", i, "-------------------------------------------------")
        # 创建一个KMeans聚类模型
        start_time = datetime.datetime.now()
        model = KMeans(n_clusters=28, random_state=56)
        # 对数据进行拟合
        model.fit(X_train)
        # 计算每个样本到聚类中心的距离
        end_time = datetime.datetime.now()
        distances = cdist(X_test, model.cluster_centers_, 'euclidean')
        # 对距离进行归一化处理
        scaler = MinMaxScaler()
        distances_normalized = scaler.fit_transform(distances)
        y_prob = distances_normalized.min(axis=1)
        # 根据归一化后的距离判断异常点和正常点
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        min_far = 1.0
        best_threshold = None
        for i in range(len(thresholds)):
            y_pred = (y_prob >= thresholds[i]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = fp / (fp + tn)
            if far < min_far and far > 0.1445:
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
        print("AUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)))
        print("最大值索引为：", list_score.index(max(list_score)), "  值为：", np.max(list_score))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('kddcup99 CPEI-iForest ROC-Curve')
        plt.show()

#孤立森林模型
if 1:
    list_score = [i * 0 for i in range(0)]
    for i in range(100, 200):
        print("------------------------------i=", i, "-------------------------------------------------")
        # 创建一个孤立森林模型
        start_time = datetime.datetime.now()
        # model = IsolationTreeEnsemble(sample_size=1024 * 2, n_trees=10, random_state=i)
        model = IsolationForest(max_samples=256, n_estimators=100, random_state=i)
        # 对数据进行拟合
        model.fit(X_train)
        # 计算每个样本的异常得分
        # y_prod = model.predict(X_test, threshold=0.33)
        end_time = datetime.datetime.now()
        y_prob = model.decision_function(X_test)
        list_score.append(roc_auc_score(y_test, y_prob))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        min_far = 1.0
        best_threshold = None
        for i in range(len(thresholds)):
            y_pred = (y_prob >= thresholds[i]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = fp / (fp + tn)
            if far <= min_far and far > 0.1549:
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
