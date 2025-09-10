import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision.utils import save_image
 
 
# 变分自编码器
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.labels = 10  # 标签数量
 
        # 编码器层
        self.fc1 = nn.Linear(input_size + self.labels, 512)  # 编码器输入层
        self.fc2 = nn.Linear(512, latent_size)
        self.fc3 = nn.Linear(512, latent_size)
 
        # 解码器层
        self.fc4 = nn.Linear(latent_size + self.labels, 512)  # 解码器输入层
        self.fc5 = nn.Linear(512, input_size)  # 解码器输出层
 
    # 编码器部分
    def encode(self, x):
        x = F.relu(self.fc1(x))  # 编码器的隐藏表示
        mu = self.fc2(x)  # 潜在空间均值
        log_var = self.fc3(x)  # 潜在空间对数方差
        return mu, log_var
 
    # 重参数化技巧
    def reparameterize(self, mu, log_var):  # 从编码器输出的均值和对数方差中采样得到潜在变量z
        std = torch.exp(0.5 * log_var)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布中采样得到随机噪声
        return mu + eps * std  # 根据重参数化公式计算潜在变量z
 
    # 解码器部分
    def decode(self, z):
        z = F.relu(self.fc4(z))  # 将潜在变量 z 解码为重构图像
        return torch.sigmoid(self.fc5(z))  # 将隐藏表示映射回输入图像大小，并应用 sigmoid 激活函数，以产生重构图像
 
    # 前向传播
    def forward(self, x, y):  # 输入图像 x，标签 y 通过编码器和解码器，得到重构图像和潜在变量的均值和对数方差
        x = torch.cat([x, y], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y], dim=1)
        return self.decode(z), mu, log_var
 
 
# 使用重构损失和 KL 散度作为损失函数
def loss_function(recon_x, x, mu, log_var):  # 参数：重构的图像、原始图像、潜在变量的均值、潜在变量的对数方差
    MSE = F.mse_loss(recon_x, x.view(-1, input_size), reduction='sum')  # 计算重构图像 recon_x 和原始图像 x 之间的均方误差
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # 计算潜在变量的KL散度
    return MSE + KLD  # 返回二进制交叉熵损失和 KLD 损失的总和作为最终的损失值
 
 
def sample_images(epoch):
    with torch.no_grad():  # 上下文管理器，确保在该上下文中不会进行梯度计算。因为在这里只是生成样本而不需要梯度
        number = 10
        # 生成标签
        sample_labels = torch.arange(10).long().to(device)  # 0-9的标签
        sample_labels_onehot = F.one_hot(sample_labels, num_classes=10).float()
        # 生成随机噪声
        sample = torch.randn(number, latent_size).to(device)  # 生成一个形状为 (64, latent_size) 的张量，其中包含从标准正态分布中采样的随机数
        sample = torch.cat([sample, sample_labels_onehot], dim=1)  # 连接图片和标签
 
        sample = model.decode(sample).cpu()  # 将随机样本输入到解码器中，解码器将其映射为图像
        save_image(sample.view(number, 1, 28, 28), f'sample{epoch}.png', nrow=int(number / 2))  # 将生成的图像保存为文件
 
 
if __name__ == '__main__':
    batch_size = 512  # 批次大小
    epochs = 60  # 学习周期
    sample_interval = 10  # 保存结果的周期
    learning_rate = 0.001  # 学习率
    input_size = 784  # 输入大小
    latent_size = 64  # 潜在变量大小
 
    # 载入 MNIST 数据集中的图片进行训练
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # 将图像转换为张量
 
    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )  # 加载 MNIST 数据集的训练集，设置路径、转换和下载为 True
 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # 创建一个数据加载器，用于加载训练数据，设置批处理大小和是否随机打乱数据
 
    # 在使用定义的 AE 类之前，有以下事情要做:
    # 配置要在哪个设备上运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # 建立 CVAE 模型并载入到 CPU 设备
    model = CVAE().to(device)
 
    # Adam 优化器，学习率
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
    # 训练
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)  # 将输入数据移动到设备（GPU 或 CPU）上
            data = data.view(-1, input_size)  # 重塑维度
 
            labels = F.one_hot(labels, num_classes=10).float().to(device)  # 转换为独热编码
            # print(labels[1])
 
            optimizer.zero_grad()  # 进行反向传播之前，需要将优化器中的梯度清零，以避免梯度的累积
 
            # 重构图像 recon_batch、潜在变量的均值 mu 和对数方差 log_var
            recon_batch, mu, log_var = model(data, labels)
 
            loss = loss_function(recon_batch, data, mu, log_var)  # 计算损失
            loss.backward()  # 计算损失相对于模型参数的梯度
            train_loss += loss.item()
 
            optimizer.step()  # 更新模型参数
 
        train_loss = train_loss / len(train_loader)  # # 计算每个周期的训练损失
        print('Epoch [{}/{}], Loss: {:.3f}'.format(epoch + 1, epochs, train_loss))
 
        # 每10次保存图像
        if (epoch + 1) % sample_interval == 0:
            sample_images(epoch + 1)
 
        # 每训练10次保存模型
        if (epoch + 1) % sample_interval == 0:
            torch.save(model.state_dict(), f'vae{epoch + 1}.pth')