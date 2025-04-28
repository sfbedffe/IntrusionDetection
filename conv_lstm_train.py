from torch import optim
import torch.nn as nn
from conv_lstm import *  # myDataset, myModel
from torch.utils.data import DataLoader
# # 超参数
# TIME_STEP = 3
# BATCH_SIZE = 10
# HIDDEN_DIM = 16  # output channel number 
# DROPOUT = 0.2
# FC_FEATURE = 200  # 1600 -> 200 -> 5
# 载入数据
data_path = r"E:\Experiments\IntrusionDetection\NSL-KDD_Prep\KDDTrain20.txt"
label_path = r"E:\Experiments\IntrusionDetection\NSL-KDD_Prep\KDDTrain20_label_multip.txt"
my_dataset =  myDataset((data_path, label_path))
my_dataloader = DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 实例化模型
my_model = myModel()
# 定义优化器
# ? 哪个优化器更适合
# optimizer = optim.Adam(my_model.parameters(), lr=LEARN_RATE, betas=[0.9, 0.999], eps=1e-8)
optimizer = optim.SGD(my_model.parameters(), lr=LEARN_RATE, momentum=0.9)
# 定义学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# 定义损失器
# ! 交叉熵损失函数期望目标张量是0维或1维的类索引，而不是独热编码形式
losser = nn.CrossEntropyLoss()  # criterion
# 模型训练
my_model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_model.to(device)
log_loss = 0.0
for epoch in range(EPOCH):
    print(f"!! Epoch: {epoch}!!")
    for index, (data, label) in enumerate(my_dataloader):
        # 前向传播
        label = torch.argmax(label, dim=1)  # 转换为索引标签
        data = data.to(device)
        label = label.to(device)
        output = my_model(data)
        loss = losser(output, label)
        
        # 反向传播
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        # ! 后期考虑梯度累加增大batch_size
        optimizer.zero_grad()  # 清除当前梯度
        
        # 记录过程
        log_loss += loss.item()
        if (index+1) % 100 == 0:
            print(f'Index [{index+1}], Loss: {log_loss/100 :.4f}')
            log_loss = 0.0
        
        if index == 1500:
            break
    scheduler.step()
        
# 模型测试
# model.eval()
    
