"""NSL-KDD数据预处理"""
import pandas as pd
import arff
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# 载入数据
file_train = "E:/Experiments/NSL_KDD/Original_NSL_KDD/KDDTrain+.txt"
file_test = "E:/Experiments/NSL_KDD/Original_NSL_KDD/KDDTest+.txt"
file_train20 = "E:/Experiments/NSL_KDD/Original_NSL_KDD/KDDTrain+_20Percent.txt"
data_train = pd.read_csv(file_train, header=None)
data_test = pd.read_csv(file_test, header=None)
data_train20 = pd.read_csv(file_train20, header=None)
# 拼接测试集与训练集，保证独热编码长度一致
data_all = pd.concat([data_train, data_test, data_train20], ignore_index=True)
data_all.columns = data_all.columns.astype(str)
print(data_all.shape)
# 分割数据和标签
data = data_all.iloc[:, :-2]
label = data_all.iloc[:, -2:-1]
print(data.head())
print(label.head())
# 初始化 OneHotEncoder
Encoder = OneHotEncoder(sparse_output=False)  # sparse_output输出密集矩阵
column_index = ['1', '2', '3']
data_onehot = Encoder.fit_transform(data[column_index])
# 将独热编码后的数据替换原始数据中的分类特征
data_onehot = pd.DataFrame(data_onehot, columns=Encoder.get_feature_names_out(column_index))
data_encoded = pd.concat([data_onehot, data.drop(columns=column_index)], axis=1)
print(data_encoded.head())
# 拆开测试集与训练集
data_encoded_train = data_encoded.iloc[:125973, :]
data_encoded_test = data_encoded.iloc[125973:125973+22544, :]
data_encoded_train20 = data_encoded.iloc[-25192:, :]
# 数据归一化
Scaler = MinMaxScaler()
scaled_data_train = Scaler.fit_transform(data_encoded_train)
scaled_data_test = Scaler.fit_transform(data_encoded_test)
scaled_data_train20 = Scaler.fit_transform(data_encoded_train20)
# print(np.max(scaled_data))
scaled_data_train = pd.DataFrame(scaled_data_train)
scaled_data_test = pd.DataFrame(scaled_data_test)
scaled_data_train20 = pd.DataFrame(scaled_data_train20)
print(scaled_data_train.shape)
print(scaled_data_test.shape)
print(scaled_data_train20.shape)

# 标签编码
CLASS = {'NORMAL':('normal'),
        'PROBE':('ipsweep','mscan','nmap','portsweep','saint','satan'),
        'DOS':('apache2','back','land','mailbomb','neptune','pod','processtable','smurf','teardrop','udpstorm'),
        'U2R':('buffer_overflow','httptunnel','loadmodule','perl','ps','rootkit','sqlattack','xterm'),
        'R2L':('ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'worm', 'xlock', 'xsnoop')}
class_key = list(CLASS.keys())
def multip_class(class_name):
    for ind, clas in enumerate(CLASS):  #先分类，再编码
        if class_name in CLASS[clas]:
            return class_key[ind]

label_binary = pd.DataFrame(np.zeros_like(label), index=label.index, columns=label.columns)
label_multip = pd.DataFrame(np.zeros_like(label), index=label.index, columns=label.columns)
label_binary['41'] = label['41'].apply(lambda class_name: "normal" if class_name == "normal" else "abnormal")
label_multip['41'] = label['41'].apply(lambda class_name: multip_class(class_name))
label_binary_encoded = pd.DataFrame(Encoder.fit_transform(label_binary))
label_multip_encoded = pd.DataFrame(Encoder.fit_transform(label_multip))
label_binary_encoded_train = label_binary_encoded.iloc[:125973, :]
label_binary_encoded_test = label_binary_encoded.iloc[125973:125973+22544, :]
label_binary_encoded_train20 = label_binary_encoded.iloc[-25192:, :]
label_multip_encoded_train = label_multip_encoded.iloc[:125973, :]
label_multip_encoded_test = label_multip_encoded.iloc[125973:125973+22544, :]
label_multip_encoded_train20 = label_multip_encoded.iloc[-25192:, :]

# 保存数据
scaled_data_train.to_csv("NSL-KDD_Prep/KDDTrain.txt", sep=",", index=False, header=False)
scaled_data_test.to_csv("NSL-KDD_Prep/KDDTest.txt", sep=",", index=False, header=False)
scaled_data_train20.to_csv("NSL-KDD_Prep/KDDTrain20.txt", sep=",", index=False, header=False)
label_binary_encoded_train.to_csv("NSL-KDD_Prep/KDDTrain_label_binary.txt", sep=",", index=False, header=False)
label_multip_encoded_train.to_csv("NSL-KDD_Prep/KDDTrain_label_multip.txt", sep=",", index=False, header=False)
label_binary_encoded_test.to_csv("NSL-KDD_Prep/KDDTest_label_binary.txt", sep=",", index=False, header=False)
label_multip_encoded_test.to_csv("NSL-KDD_Prep/KDDTest_label_multip.txt", sep=",", index=False, header=False)
label_binary_encoded_train20.to_csv("NSL-KDD_Prep/KDDTrain20_label_binary.txt", sep=",", index=False, header=False)
label_multip_encoded_train20.to_csv("NSL-KDD_Prep/KDDTrain20_label_multip.txt", sep=",", index=False, header=False)