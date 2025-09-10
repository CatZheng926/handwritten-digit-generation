import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

# 定义 CVAE 类（与之前的模型定义相同）
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.labels = 10  # 标签数量

        # 编码器层
        self.fc1 = nn.Linear(784 + self.labels, 512)  # 输入大小784 + 标签数量
        self.fc2 = nn.Linear(512, 64)  # 潜在空间的均值
        self.fc3 = nn.Linear(512, 64)  # 潜在空间的对数方差

        # 解码器层
        self.fc4 = nn.Linear(64 + self.labels, 512)  # 解码器输入层
        self.fc5 = nn.Linear(512, 784)  # 输出层，大小为784（28x28图像的展平）

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
        x = torch.cat([x, y], dim=1)  # 合并图像和标签
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y], dim=1)  # 将潜在变量和标签再次合并
        return self.decode(z), mu, log_var

# 配置参数
input_size = 784  # 输入大小
latent_size = 64  # 潜在空间大小
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'vae30.pth'  # 保存的模型路径

# 加载训练好的模型
model = CVAE().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置为评估模式

# 生成指定数字图像的函数
def generate_image(label):
    # 创建一个包含指定标签的独热编码（one-hot encoding）
    label_onehot = F.one_hot(torch.tensor([label]), num_classes=10).float().to(device)
    
    # 生成随机潜在变量
    z = torch.randn(1, latent_size).to(device)
    
    # 将潜在变量与标签连接起来
    z = torch.cat([z, label_onehot], dim=1)
    
    # 使用解码器生成图像
    with torch.no_grad():  # 不需要计算梯度
        generated_image = model.decode(z).cpu()  # 生成图像
    
    # 保存生成的图像
    save_image(generated_image.view(1, 1, 28, 28), f"generated_label_{label}.png")
    print(f"Image for label {label} saved!")

# 使用指定的标签生成图像
if __name__ == "__main__":
    label_to_generate = 9  # 你想生成的数字（例如，数字 9）
    generate_image(label_to_generate)
