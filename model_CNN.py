import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt


# 参数初始化
def gaussian_weights_init(m):#自定义的权重初始化函数 `gaussian_weights_init`，用于对卷积层的权重进行高斯初始化。
    classname = m.__class__.__name__#获取输入参数 `m` 的类名，获取类对象的类名，即返回一个表示类名的字符串。
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:#：判断类名中是否包含字符串 'Conv'，即判断是否是卷积层。
        m.weight.data.normal_(0.0, 0.04)# `m.weight.data`：表示卷积层的权重参数，normal_(0.0, 0.04)`：使用PyTorch 的`normal_` 方法，给权重参数赋予服从均值为 0.0，标准差为 0.04 的正态分布随机值
#这段代码的作用是，在使用该函数对模型的权重进行初始化时，会遍历模型的每个模块（如卷积层），并对卷积层的权重参数进行高斯初始化


# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)#使用模型将数据传递到前向传播过程中，计算预测值。
        #print(pred.data)
        pred = np.argmax(pred.data.numpy(), axis=1)#模型的预测结果是一个概率分布或得分，通过找到最大值的索引来确定最终的预测类别。返回一个一维数组，其中每个元素是对应行中最大元素的索引。`pred.data.numpy()` 将预测结果的数据转换为 NumPy 数组,`np.argmax()` 是 NumPy 库中的一个函数，用于在数组中查找最大元素的索引,`pred.data.numpy()` 返回的 NumPy 数组是多维的，`axis=1` 参数表示在第1个轴上进行操作，即在每一行中找到最大元素的索引。
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc

# 我们通过继承Dataset类来创建我们自己的数据加载类，命名为FaceDataset
class FaceDataset(data.Dataset):
    '''
    首先要做的是类的初始化。之前的image-emotion对照表已经创建完毕，
    在加载数据时需用到其中的信息。因此在初始化过程中，我们需要完成对image-emotion对照表中数据的读取工作。
    通过pandas库读取数据，随后将读取到的数据放入list或numpy中，方便后期索引。
    '''
    # 初始化
    def __init__(self, root):#root表示数据集的根目录
        super(FaceDataset, self).__init__()#调用父类的构造方法
        self.root = root#将输入的根目录赋值给数据集对象的 `root` 属性
        df_path = pd.read_csv(root + '\\image_emotion2.csv', header=None, usecols=[0])#指定只读取第一列的数据
        df_label = pd.read_csv(root + '\\image_emotion2.csv', header=None, usecols=[1])#指定只读取第二列的数据
        self.path = np.array(df_path)[:, 0]#将 `df_path` 转换为 NumPy 数组，并提取第一列的数据，存储在数据集对象的 `path` 属性中。
        self.label = np.array(df_label)[:, 0]

    '''
    接着就要重写getitem()函数了，该函数的功能是加载数据。
    在前面的初始化部分，我们已经获取了所有图片的地址，在这个函数中，我们就要通过地址来读取数据。
    由于是读取图片数据，因此仍然借助opencv库。
    需要注意的是，之前可视化数据部分将像素值恢复为人脸图片并保存，得到的是3通道的灰色图（每个通道都完全一样），
    而在这里我们只需要用到单通道，因此在图片读取过程中，即使原图本来就是灰色的，但我们还是要加入参数从cv2.COLOR_BGR2GARY，
    保证读出来的数据是单通道的。读取出来之后，可以考虑进行一些基本的图像处理操作，
    如通过高斯模糊降噪、通过直方图均衡化来增强图像等（经试验证明，在本项目中，直方图均衡化并没有什么卵用，而高斯降噪甚至会降低正确率，可能是因为图片分辨率本来就较低，模糊后基本上什么都看不清了吧）。
    读出的数据是48X48的，而后续卷积神经网络中nn.Conv2d() API所接受的数据格式是(batch_size, channel, width, higth)，本次图片通道为1，因此我们要将48X48 reshape为1X48X48。
    '''

    # 读取某幅图片和对应的标签，item为索引号
    #对图片进行预处理，使返回的数据可直接用于训练神经网络
    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])
        faces=cv2.resize(face,(48,48))
        # 读取单通道灰度图
        face_gray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)#只需要用到单通道，因此在图片读取过程中，即使原图本来就是灰色的，但我们还是要加入参数从cv2.COLOR_BGR2GARY
        # 高斯模糊
        # face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)
        # 直方图均衡化,将图像的灰度值分布调整得更均匀。这可以帮助增强图像的对比度和细节。
        face_hist = cv2.equalizeHist(face_gray)
        # 像素值标准化,使用 `reshape` 方法将图像的形状调整为 `(1, 48, 48)`，以适配后续的卷积神经网络的输入要求。然后将图像的像素值除以 255.0，将像素值范围缩放到 0 到 1 之间
        #faces = face.resize(48, 48)

        face_normalized = face_hist.reshape(1, 48, 48) / 255.0 # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
        # 用于训练的数据需为tensor类型
        face_tensor = torch.from_numpy(face_normalized) # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
        face_tensor = face_tensor.type('torch.FloatTensor') # 将张量数据类型指定为 `'torch.FloatTensor'`。这是为了确保数据类型与模型的输入要求相匹配，因为有些模型要求输入的数据类型必须是浮点型。
        label = self.label[item]
        return face_tensor, label#最终返回的数据可以直接用于训练神经网络模型


    '''
    最后就是重写len()函数获取数据集大小了。
    self.path中存储着所有的图片名，获取self.path第一维的大小，即为数据集的大小。
    '''
    # 获取数据集样本个数
    def __len__(self):
        return self.path.shape[0]#self.path中存储着所有的图片名，获取self.path第一维的大小，即为数据集的大小。



class FaceCNN(nn.Module):#继承自 `nn.Module` 的子类需要实现 `__init__()` 和 `forward()` 方法
    # 初始化网络结构
    def __init__(self):
        super(FaceCNN, self).__init__()

        # 第一次卷积、池化
        self.conv1 = nn.Sequential(#`nn.Sequential()`是PyTorch中的一个容器，用于顺序地组合多个神经网络层,通过将多个网络层封装在`nn.Sequential()`中，可以直接调用整个容器对象来进行前向传播，无需逐层调用。容器中每个网络层的输出会自动作为下一层的输入
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 1, 48, 48), output:(bitch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # 卷积层,通过使用`nn.Conv2d`，可以定义一个二维卷积层，并将其作为神经网络的一部分。在神经网络的前向传播过程中，输入数据会经过该卷积层进行特征提取和卷积操作，生成输出特征图。需要注意的是，`nn.Conv2d`只是卷积层的定义，实际的卷积操作是在神经网络的前向传播过程中进行的。
            nn.BatchNorm2d(num_features=64), # 归一化,num_features=64`：表示输入数据的特征通道数，即输入数据的深度。nn.BatchNorm2d`是PyTorch中用于定义二维批归一化层的类。nn.BatchNorm2d`的主要作用是对每个特征通道上的数据进行标准化，使其具有零均值和单位方差。这有助于缓解梯度消失和梯度爆炸问题，并加速模型的训练。此外，批归一化还具有一定的正则化效果，可以减少模型的过拟合。
            nn.RReLU(inplace=True), # 激活函数,inplace=True`：表示是否进行原地操作（in-place operation），即是否直接在输入张量上进行修改。RReLU激活函数是一种类似于ReLU激活函数的变种，它在负半轴上引入了随机性。RReLU(x) = max(x, a*x)其中a是从均匀分布[lower, upper]中随机采样得到的一个值，用于引入随机性。通常情况下，a的范围是[0.125, 0.333]。
            # output(bitch_size, 64, 24, 24)
            nn.MaxPool2d(kernel_size=2, stride=2), # 最大值池化
        )

        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            # input:(bitch_size, 64, 24, 24), output:(bitch_size, 128, 24, 24), (24-3+2*1)/1+1 = 24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 128, 12 ,12)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积、池化
        self.conv3 = nn.Sequential(
            # input:(bitch_size, 128, 12, 12), output:(bitch_size, 256, 12, 12), (12-3+2*1)/1+1 = 12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 256, 6 ,6)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)#`apply()`方法的作用是将一个函数应用于模型的所有子模块，包括当前模块本身。具体而言，它会遍历模型的所有子模块，并将指定的函数应用于每个子模块。这里的指定函数是gaussian_weights_init
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # 全连接层
        #需要注意的是，在全连接层中，`in_features`和`out_features`参数分别表示输入和输出特征的维度。这些参数的选择通常根据具体任务和数据的特点进行调整。
        self.fc = nn.Sequential(#Fully Connected Layer，用于将卷积层提取的特征进行分类或回归任务。
            nn.Dropout(p=0.2),#用于防止过拟合的正则化操作。它以概率`p=0.2`随机将输入的部分元素置为0，以减少神经网络的复杂性。
            nn.Linear(in_features=256*6*6, out_features=4096),#`nn.Linear(in_features=256*6*6, out_features=4096)`：这是一个线性变换层，将输入特征的维度从`256*6*6`映射到`4096`。`in_features`表示输入特征的维度，`out_features`表示输出特征的维度。在pytorch中的nn.Linear表示线性变换，官方文档给出的数学计算公式是y=xAT+b其中x是输入，A是权值，b是偏置，y是输出，
            nn.RReLU(inplace=True),#nn.Linear()`是PyTorch中的一个类，用于定义全连接层
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=8),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 数据扁平化
        x = x.view(x.shape[0], -1)#，将卷积层的输出特征从多维张量转换为一维向量，以便传递给全连接层进行进一步处理。`x.shape[0]` 保持不变，作为批量大小。- `-1` 表示自动计算第二维度的大小，以保持张量中的元素总数不变。
        y = self.fc(x)
        return y

def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):#wt_decay为权重衰减
    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    model = FaceCNN()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)#model.parameters()` 获取模型的可训练参数，创建优化器对象，optim.SGD` 是 PyTorch 库中的一个优化器类，
    # 学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)#每经过step_size 个epoch，做一次学习率decay，以gamma值为缩小倍数。
    # 逐轮训练
    #用于存储损失值loss,精度acc
    Loss_list = []
    Acc_train_list=[]
    Acc_val_list=[]
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        #scheduler.step() # 学习率衰减
        model.train() # 用于将模型设置为训练模式，当模型处于训练模式时，模型会启用一些特定的操作，例如DropOut和Batch Normalization
        for images, emotion in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model.forward(images)
            # 误差计算
            loss_rate = loss_function(output, emotion)
            # 误差的反向传播
            loss_rate.backward()#根据损失值计算模型参数的梯度,`backward()` 是一个函数，用于计算 `loss_rate` 相对于模型中所有可训练参数的梯度反向传播算法使用链式法则来计算梯度。它从损失函数开始，沿着计算图向后传播，依次计算每个操作对于损失函数的梯度，最终得到模型参数的梯度。
            # 更新参数
            optimizer.step()#根据梯度更新模型的参数。在调用 `optimizer.step()` 后，优化器会根据之前计算得到的梯度信息来更新模型的参数。parameter = parameter - learning_rate * gradient`，其中 `parameter` 是模型的一个可训练参数，`learning_rate` 是学习率，`gradient` 是该参数的梯度。
        scheduler.step()  # 学习率衰减
        # 打印每轮的损失
        print('After {} epochs , the loss_rate is : '.format(epoch+1), loss_rate.item())
        # 将每次训练后的准确率或损失值存入列表中
        Loss_list.append(loss_rate)
        if epoch % 5 == 0:
            model.eval() # 模型评估, 在评估模式下，模型的行为与推理阶段一致，即使用训练好的参数进行预测，而不进行参数更新。在评估模式下，模型使用训练好的参数进行预测，而不进行参数更新。
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('After {} epochs , the acc_train is : '.format(epoch+1), acc_train)
            Acc_train_list.append(acc_train)
            print('After {} epochs , the acc_val is : '.format(epoch+1), acc_val)
            Acc_val_list.append(acc_val)
    # 对存入列表中的数据进行强制类型转换
    for i in range(0, len(Loss_list)):
        Loss_list[i] = float(Loss_list[i])
    for i in range(0,len(Acc_val_list)):
        Acc_train_list[i]=float(Acc_train_list[i]*100)
        Acc_val_list[i]=float(Acc_val_list[i]*100)
    x1=range(0,len(Acc_train_list))
    x2=range(0,len(Acc_val_list))
    x3 = range(0, len(Loss_list))
    y1=Acc_train_list
    y2=Acc_val_list
    y3 = Loss_list

    #Acc_train图像
    plt.subplot(2, 1, 2)
    plt.plot(x1, y1, 'o-')
    plt.plot(x2, y2, 'o-')
    plt.title('model')
    plt.ylabel('accuracy :%')
    plt.xlabel('epoch')
    my_yTicks1 = np.arange(20, 110, 10)
    plt.yticks(my_yTicks1)

    '''# Acc_test图像
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'o-')
    plt.title('model')
    plt.ylabel('test_accuracy :%')
    plt.xlabel('epoch')
    my_yTicks1 = np.arange(20, 110, 10)
    plt.yticks(my_yTicks1)'''

    # 绘制损失率图像
    plt.subplot(2, 1, 1)
    plt.plot(x3, y3, '.-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    my_yTicks2 = np.arange(0.00, 1.6, 0.1)
    plt.yticks(my_yTicks2)
    # 输出并保存图像
    plt.savefig("templates/loss8.jpg")
    plt.show()
    return model

def main():
    # 数据集实例化(创建数据集)
    train_dataset = FaceDataset(root='D:/facial recognition/final_dataset3/train')
    val_dataset = FaceDataset(root='D:/facial recognition/final_dataset3/test')
    # 超参数可自行指定
    model = train(train_dataset, val_dataset, batch_size=128, epochs=86, learning_rate=0.02, wt_decay=0)
    # 保存模型参数
    torch.save(model, 'model/model_cnn8.pkl')


if __name__ == '__main__':
    main() 