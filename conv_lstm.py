import torch
import torch.nn as nn
import torch.nn.functional as F
from Help.convlstm import ConvLSTM
    
if __name__ == '__main__':
    # 创建一个示例输入张量，假设输入数据的形状为 (batch_size, seq_len, channels, height, width)
    batch_size = 2
    seq_len = 10
    channels = 3
    height = 64
    width = 64
    input_tensor = torch.randn(batch_size, seq_len, channels, height, width)
    # 初始化 ConvLSTM 模型
    convlstm = ConvLSTM(input_dim=channels, hidden_dim=16, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True)
    # 前向传播
    layer_output_list, last_state_list = convlstm(input_tensor)
    # 获取最后一层的隐藏状态
    h = last_state_list[0][0]  # 0 表示层索引，0 表示 h 索引
    print("最后一层的隐藏状态形状：", h.shape)