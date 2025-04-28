import torch
import torch.nn as nn
import torch.nn.functional as F
from Help.convlstm import ConvLSTM
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

"""重写Dataset类"""
class myDataset(Dataset):
    def __init__(self, file_path:tuple[str]):
        super().__init__()
        self.all_data = pd.read_csv(file_path[0], header=None)
        self.label = pd.read_csv(file_path[1], header=None)
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        a_data = self.all_data.loc[index].tolist()
        a_data = np.append(np.asarray(a_data), np.zeros(22))  
        a_data = np.reshape(a_data, (1, 12, 12))
        a_data = torch.tensor(a_data.astype(np.float32))  # 匹配模型偏置float32类型
        a_label = self.label.loc[index].tolist()
        a_label = torch.tensor(np.asarray(a_label, dtype=np.float32))
        return a_data, a_label


"""定义Reshape类"""
class Reshape(nn.Module):
    def __init__(self, batch_size, time_step, channel, hight, width):
        super(Reshape, self).__init__()
        self.shape = (batch_size, time_step, channel, hight, width)
    def forward(self, x):
        return x.view(self.shape)

"""超参数"""
TIME_STEP = 8
BATCH_SIZE = 16
HIDDEN_DIM = 32  # output channel number 
NUM_LAYER = 3
DROPOUT = 0.2
FC_FEATURE = 200  # 1600 -> 80 -> 5
LEARN_RATE = 0.01
EPOCH = 60

"""模型结构"""
class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ts_feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=TIME_STEP, kernel_size=(5,5), padding=0, stride=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=TIME_STEP),
            # ! 构造伪时序 C0 -> T ：(B0, C0, H0, W0) ==> (B0, T, 1, H0, W0)  
            Reshape(batch_size=BATCH_SIZE, time_step=TIME_STEP, channel=1, hight=8, width=8),
            # ! LSTM输出为(ci, hi)元组，hi和ci形状为(L, B, C, W, H), 只输出最后一个时间步，且W、H维度不变
            ConvLSTM(input_dim=1, hidden_dim=HIDDEN_DIM, kernel_size=(3,3), num_layers=NUM_LAYER, batch_first=True)
            # ? ConvLSTM内部卷积后是否需要需要Batch-Norm
        )

        self.classfier = nn.Sequential(
            nn.Flatten(start_dim=1),  # 首维为0，ConvLSTM输出(B, C, W, H)
            # ? 中间层神经元数多少合适
            nn.Linear(in_features=8*8*HIDDEN_DIM, out_features=FC_FEATURE),
            nn.ReLU(inplace=True),
            # nn.Dropout(DROPOUT),
            nn.Linear(in_features=FC_FEATURE, out_features=5)
            # nn.Softmax(dim=-1)
        )


    def forward(self, x):
        _, h = self.ts_feature(x)
        x = self.classfier(h[0][0])
        return x


if __name__ == '__main__':
    """测试myModule"""
    model = myModel()
    input = torch.rand((BATCH_SIZE, 1, 12, 12))
    output = model(input)
    print(output.shape)
    """测试myDataset"""
    # data_path = r"E:\Experiments\IntrusionDetection\NSL-KDD_Prep\KDDTrain.txt"
    # label_path = r"E:\Experiments\IntrusionDetection\NSL-KDD_Prep\KDDTrain_label_binary.txt"
    # my_dataset =  myDataset((data_path, label_path))
    # my_dataloader = DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE)
    # for i, d in enumerate(my_dataloader):
    #     print(d)
    #     if i == 3:
    #         break
    """测试Reshape()类"""
    # arr1 = np.zeros((3,3))
    # arr2 = np.ones((3,3))
    # arr3 = 2 * np.ones((3,3))
    # arr = np.stack((arr1, arr2, arr3))
    # print(arr)
    # arr_re = np.reshape(arr, (1,3,3,3))
    # print(arr_re)
    # reshape = Reshape(batch_size=1, time_step=3, channel=1, hight=3, width=3)
    # arr_re_re = reshape(torch.tensor(arr_re))
    # print(arr_re_re)