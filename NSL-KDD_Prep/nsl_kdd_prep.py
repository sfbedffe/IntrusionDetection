"""NSL-KDD数据预处理"""
import pandas as pd
import arff
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def get_nsl_kdd(data_set):
    # 载入数据
    file_path = "E:/Experiments/IntrusionDetection/NSL_KDD/KDD"+data_set+"+.txt"
    data_all = pd.read_csv(file_path, header=None)
    data_all.columns = data_all.columns.astype(str)
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

    # 数据归一化
    Scaler = MinMaxScaler()
    scaled_data = Scaler.fit_transform(data_encoded)
    print(np.max(scaled_data))
    scaled_data = pd.DataFrame(scaled_data)


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
    print(label_binary_encoded)
    print(label_multip_encoded)

    scaled_data.to_csv("NSL-KDD_Prep/KDD"+data_set+"_data.txt", sep=",", index=False, header=False)
    label_binary_encoded.to_csv("NSL-KDD_Prep/KDD"+data_set+"_label_onehot_binary.txt", sep=",", index=False, header=False)
    label_multip_encoded.to_csv("NSL-KDD_Prep/KDD"+data_set+"_label_onehot_multip.txt", sep=",", index=False, header=False)

if __name__ == "__main__":
    get_nsl_kdd("Train")
    get_nsl_kdd("Test")
    print("over!")