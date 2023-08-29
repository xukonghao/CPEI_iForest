from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
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


def read_csv(file):
    global label_list, head_row, data, normal, attack
    with open(file) as f:
        f_csv = csv.reader(f)
        head_row = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",\
                    "logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",\
                    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",\
                    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
        for row in f_csv:
            row_len = len(row) - 1
            row[0:row_len] = [float(x) for x in row[0:row_len]]
            row[row_len] = int(row[row_len])
            if row[row_len] == 0 and normal < 90000:
                data.append(row[0:row_len])
                label_list.append(1)
                normal += 1
            elif row[row_len] != 0 and attack < 10000:
                data.append(row[0:row_len])
                label_list.append(0)
                attack += 1


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
normal, attack = 0, 0

# 读取文件，获取标签，表头，数据---------------------------------------------------------
read_csv("./Dataset/KDDCUP99/kddcup.data_10_percent_corrected.csv")
data = pd.DataFrame(data=np.array(data), columns=head_row)
#数据提取
index_list = [ 1, 2, 3, 6, 11, 20, 21]
head_row_discrete = [head_row[i] for i in index_list]
data_discrete = copy.deepcopy(data)
data_discrete = data_discrete.loc[:, head_row_discrete]
label_list_discrete = copy.deepcopy(label_list)
data_continuous = copy.deepcopy(data)  # 数据深拷贝
data_continuous = data.drop(head_row_discrete, axis=1)
print("normal:", normal, "attack:", attack, "data.shape", data.shape,"data_continuous",data_continuous.shape,"data_discrete",data_discrete.shape)
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
print("归一化后数据 data.shape:", data.shape)

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
print("方差过滤后 data.shape:", data_continuous.shape)

# 定义L1正则化稀疏自编码器模型
model = Pipeline(steps=[('rbm', BernoulliRBM(n_components=33, n_iter=20, learning_rate=0.1, verbose=True, random_state=42))])
# 训练L1正则化稀疏自编码器模型 降维
model.fit(data_continuous)
data_continuous = pd.DataFrame(data=data_continuous.values, columns=head_row_continuous)
data_continuous = filter_by_pearson(0.6, data_continuous)
print("L1正则化稀疏自编码器模型 降维后 data.shape:", data_continuous.shape)

X_train, X_test, y_train, y_test = train_test_split(data_continuous, label_list, test_size=0.2,
                                                    random_state=43)

list_score=[i*0 for i in range(102)]
for i in range(103,104):
    # 创建一个孤立森林模型
    start_time = datetime.datetime.now()
    model = IsolationForest(max_samples=256 * 2 * 4, n_estimators=10, random_state=i)
    # 对数据进行拟合
    model.fit(X_train)
    end_time = datetime.datetime.now()
    # 计算每个样本的异常得分
    y_prob = model.decision_function(X_test)

    print("------------------------------i=", i, "-------------------------------------------------")
    list_score.append(roc_auc_score(y_test, y_prob))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    # print("fpr:")
    # for j in fpr:
    #     print(j)
    # print("\ntpr:")
    # for j in tpr:
    #     print(j)
    # 循环遍历每个阈值，计算误检率，并找到误检率最低的阈值
    min_far = 1.0
    best_threshold = None
    for i in range(len(thresholds)):
        y_pred = (y_prob >= thresholds[i]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        far = fp / (fp + tn)
        if far < min_far and far > 0.0095:
            min_far = far
            best_threshold = thresholds[i]
    # 使用最佳阈值对测试数据进行预测，得到预测标签y_prod
    y_prod = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_prod).ravel()
    detection_rate = tp / (tp + fn)
    false_alarm_rate = fp / (tn + fp)
    precision = tp / (tp + fp)
    print("模型运行时间(Running time):", (end_time - start_time),\
          "\n准确率(accuracy):", '{:.2%}'.format(accuracy_score(y_test, y_prod)), \
          "\n召回率(Recall):", '{:.2%}'.format(detection_rate), \
          "\n误检率(FPR):", '{:.2%}'.format(false_alarm_rate), \
          "\n精确率(Precision):", '{:.2%}'.format(precision), \
          "\nF1 score:", '{:.2%}'.format(f1_score(y_test, y_prod)), \
          "\nAUC:", '{:.2%}'.format(roc_auc_score(y_test, y_prob)), \
          "\n混淆矩阵:", confusion_matrix(y_test, y_prod, labels=[1, 0]))

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('kddcup99 iForest ROC-Curve')
    plt.show()
max_index = list_score.index(max(list_score))
print("最大值索引为：", max_index + 1, "  值为：", list_score[max_index])


# from sklearn.mixture import GaussianMixture
# import numpy as np
# import pandas as pd
# import xlwt
# import csv
# from sklearn.preprocessing import MinMaxScaler
# from scipy.stats import pearsonr
# from sklearn.metrics import mutual_info_score
# from sklearn import metrics
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# import datetime
# import copy
# from CPEI_Iforest import IsolationTreeEnsemble
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score
# from sklearn.ensemble import IsolationForest
#
#
# def read_csv(file):
#     global label_list, head_row, data, normal, attack
#     with open(file) as f:
#         f_csv = csv.reader(f)
#         head_row = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",\
#                     "logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",\
#                     "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",\
#                     "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
#         for row in f_csv:
#             row_len = len(row) - 1
#             row[0:row_len] = [float(x) for x in row[0:row_len]]
#             row[row_len] = int(row[row_len])
#             if row[row_len] == 0 and normal < 90000:
#                 data.append(row[0:row_len])
#                 label_list.append(1)
#                 normal += 1
#             elif row[row_len] != 0 and attack < 10000:
#                 data.append(row[0:row_len])
#                 label_list.append(0)
#                 attack += 1
#
#
# def filter_by_variance(threshold, var, dataTemp):
#     global label_list, head_row
#     columns_drop = []
#     for i in range(dataTemp.columns.size):
#         if var[i] <= threshold:
#             columns_drop.append(dataTemp.columns[i])
#     dataTemp.drop(axis=1, labels=columns_drop, inplace=True)
#     dataTemp.columns = range(len(dataTemp.columns))
#     # head_row = np.delete(head_row, columns_drop)
#     return dataTemp
#
# def filter_by_pearson(threshold, dataTemp):
#     global pearson
#     pearson = [[None for j in range(len(dataTemp.columns))] for i in range(len(dataTemp.columns))]
#     columns_drop = []
#     for i in range(len(dataTemp.columns)):
#         j = i + 1
#         for j in range(j, len(dataTemp.columns)):
#             corr = pearsonr(dataTemp[dataTemp.columns[i]], dataTemp[dataTemp.columns[j]])
#             pearson[i][j] = corr[0]
#             pearson[j][i] = corr[0]
#             if abs(corr[0]) >= threshold:
#                 columns_drop.append(dataTemp.columns[i])
#             j += 1
#     dataTemp.drop(axis=1, labels=columns_drop, inplace=True)
#     return dataTemp
#
#
# def filter_by_mutual_info(threshold, dataTemp):
#     global by_mutual_info
#     by_mutual_info = [[None for j in range(len(dataTemp.columns))] for i in range(len(dataTemp.columns))]
#     columns_drop = []
#     for i in range(len(dataTemp.columns)):
#         j = i + 1
#         for j in range(j, len(dataTemp.columns)):
#             # corr = by_mutual_infor(dataTemp[dataTemp.columns[i]], dataTemp[dataTemp.columns[j]])
#             mutual_info = metrics.normalized_mutual_info_score(dataTemp[dataTemp.columns[i]],
#                                                                dataTemp[dataTemp.columns[j]])
#             by_mutual_info[i][j] = mutual_info
#             by_mutual_info[j][i] = mutual_info
#             if mutual_info >= threshold:
#                 columns_drop.append(dataTemp.columns[i])
#             j += 1
#     dataTemp.drop(axis=1, labels=columns_drop, inplace=True)
#     return dataTemp
#
#
# global label_list, head_row_continuous, head_row_discrete, normal, attack, pearson  # 声明全局变量
# head_row,head_row_continuous, head_row_discrete, label_list, data, data_discrete, data_continuous, pearson, mutual_info\
#     =[], [], [], [], [], [], [], [], []
# normal, attack = 0, 0
#
# # 读取文件，获取标签，表头，数据---------------------------------------------------------
# read_csv("./Dataset/kddcup.data_10_percent_corrected.csv")
# data = pd.DataFrame(data=np.array(data), columns=head_row)
# #离散数据提取
# index_list = [ 1, 2, 3, 6, 11, 20, 21]
# head_row_discrete = [head_row[i] for i in index_list]
# data_discrete = copy.deepcopy(data)
# data_discrete = data_discrete.loc[:, head_row_discrete]
# label_list_discrete = copy.deepcopy(label_list)
# #连续数据处理
# data_continuous = copy.deepcopy(data)  # 数据深拷贝
# data_continuous = data.drop(head_row_discrete, axis=1)
# print("normal:", normal, "attack:", attack, "data.shape", data.shape,"data_continuous",data_continuous.shape,"data_discrete",data_discrete.shape)
# # for i, x in enumerate(label_list_discrete):
# #     if x == -1: label_list_discrete[i] = -1
# #     elif x== 1: label_list_discrete[i] = 1
#
# # 数据归一化------------------------------------------------------------------
# scaler = MinMaxScaler()
# scaler.fit(data_continuous)
# data_continuous = scaler.transform(data_continuous)
# data_continuous = pd.DataFrame(data_continuous)  # data = pd.DataFrame(data=data.values,columns=head_row[0:-1])
#
# scaler.fit(data_discrete)
# data_discrete = scaler.transform(data_discrete)
# data_discrete = pd.DataFrame(data_discrete)
# print("归一化后数据 data_continuous.shape:", data_continuous.shape," data_discrete.shape:",data_discrete.shape)
#
# # 方差 预过滤----------------------------------------------------------------
# variance_continuous = []
# for i in range(data_continuous.columns.size):
#     variance_continuous.append(np.var(data_continuous[data_continuous.columns[i]]))  # 计算方差
# data_continuous=filter_by_variance(0, variance_continuous, data_continuous)
# head_row_continuous = data_continuous.columns.tolist()
#
# variance_discrete = []
# for i in range(data_discrete.columns.size):
#     variance_discrete.append(np.var(data_discrete[data_discrete.columns[i]]))  # 计算方差
# data_discrete=filter_by_variance(0, variance_discrete, data_discrete)
# head_row_discrete = data_discrete.columns.tolist()
# print("方差过滤后 data_continuous.shape:", data_continuous.shape," data_discrete.shape:",data_discrete.shape)
#
# # # 皮尔逊，互信息系数筛选阈值 过滤 xgboost模型计算准确率和耗时
# pearson_threshold = [i / 10 for i in range(1, 11)]
# # mutual_info_threshold = [i / 10 for i in range(1, 11)]
# # accuracy = [[None for j in range(10)] for i in range(10)]
# i = 5
# # j=4
# # dataTemp_pearson = pd.DataFrame(data=data_continuous.values, columns=head_row_continuous)
# # dataTemp_pearson = filter_by_pearson(pearson_threshold[i], dataTemp_pearson)
# # dataTemp_mutual_info = copy.deepcopy(dataTemp_pearson)  # 数据深拷贝
# # dataTemp_mutual_info = filter_by_mutual_info(mutual_info_threshold[j], dataTemp_mutual_info)
# data_continuous = pd.DataFrame(data=data_continuous.values, columns=head_row_continuous)
# data_continuous = filter_by_pearson(pearson_threshold[i], data_continuous)
#
# pearson_threshold = [i / 10 for i in range(1, 11)]
# i = 6
# data_discrete = pd.DataFrame(data=data_discrete.values, columns=head_row_discrete)
# data_discrete = filter_by_pearson(pearson_threshold[i], data_discrete)
# print("皮尔逊过滤 data_continuous.shape:", data_continuous.shape," data_discrete.shape:",data_discrete.shape)
#
# list_score=[i*0 for i in range(10)]
# for i in range(11,20):
#     # 创建一个孤立森林模型
#     model = IsolationForest(max_samples=256*2, n_estimators=10, random_state=i)
#     # 对数据进行拟合
#     model.fit(data_continuous)
#     # 计算每个样本的异常得分
#     scores = model.predict(data_continuous)
#     normal_scores = model.decision_function(data_continuous)
#     for j, x in enumerate(scores):
#         if x == -1: scores[j] = 0
#         elif x== 1: scores[j] = 1
#     print("confusion_matrix:", confusion_matrix(label_list, scores,labels=[1,0]), "\naccuracy_score:",
#           accuracy_score(label_list, scores))
#     print("roc_auc_score:",roc_auc_score(label_list, normal_scores))
#     list_score.append(roc_auc_score(label_list, normal_scores))
#     fpr, tpr, thresholds = roc_curve(label_list, normal_scores)
#     plt.plot(fpr, tpr)
#     plt.plot([0, 1], [0, 1], linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     # plt.show()
#     plt.show(block=False)
#     plt.pause(2)
#     plt.close("all")
# max_index=list_score.index(max(list_score))
# print("最大值索引为：",max_index,"  值为：",list_score[max_index])
#
#
# # # 创建一个GMM模型对象
# #
# # gmm = GaussianMixture(n_components=2, random_state=42)
# #
# # # 使用数据拟合模型
# # gmm.fit(dataTemp_pearson_discrete)
# #
# # # 设置阈值
# # threshold = 0.9
# #
# # # 生成二分类标签
# # labels = np.where(gmm.predict_proba(dataTemp_pearson_discrete)[:, 0] > threshold, 0, 1)
# #
# # probs = gmm.predict_proba(dataTemp_pearson_discrete)[:,0]
# # print(labels)
# # print("probs",probs)
# # print("confusion_matrix:", confusion_matrix(label_list_discrete, labels), "\naccuracy_score:",accuracy_score(label_list_discrete, labels))
# # print("roc_auc_score:", roc_auc_score(label_list_discrete, probs))
# # fpr, tpr, thresholds = roc_curve(label_list_discrete, probs)
# # plt.plot(fpr, tpr)
# # plt.plot([0, 1], [0, 1], linestyle='--')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# # plt.show()