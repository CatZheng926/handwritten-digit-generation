# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:26:17 2020
CGAN生成MNIST数据
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
image_size = 28 * 28
hidden_size = 256
latent_size = 128
epochs = 128
lr = 2e-4
num_classes = 10  # 0-9

# Discriminator
class D(nn.Module):
    def __init__(self, image_size, hidden_size, num_classes):
        super(D, self).__init__()
        self.layer1 = nn.Linear(image_size + num_classes, hidden_size)
        self.actF1 = nn.LeakyReLU(0.2)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.actF2 = nn.Sigmoid()
        
    def forward(self, images, labels):
        labels = nn.functional.one_hot(labels, num_classes=num_classes).float()
        x = torch.cat((images, labels), dim=1)
        x = self.layer1(x)
        x = self.actF1(x)
        x = self.layer2(x)
        x = self.actF1(x)
        x = self.layer3(x)
        y = self.actF2(x)
        return y

# Generator
class G(nn.Module):
    def __init__(self, latent_size, hidden_size, image_size, num_classes):
        super(G, self).__init__()
        self.layer1 = nn.Linear(latent_size + num_classes, hidden_size)
        self.actF1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, image_size)
        self.actF2 = nn.Tanh()
        
    def forward(self, latents, labels):
        labels = nn.functional.one_hot(labels, num_classes=num_classes).float()
        x = torch.cat((latents, labels), dim=1)
        x = self.layer1(x)
        x = self.actF1(x)
        x = self.layer2(x)
        x = self.actF1(x)
        x = self.layer3(x)
        images = self.actF2(x)
        return images

def save_generated_images(epoch):
    with torch.no_grad():
        z = torch.randn(64, latent_size).to(device)
        labels = torch.randint(0, num_classes, (64,)).to(device)
        fake_images = G(z, labels).view(-1, 1, 28, 28).cpu()
        grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(f'generated_epoch_{epoch}.png')
        plt.close()

# 图片预处理方法
transform = transforms.Compose([transforms.ToTensor()])
# 下载MNIST中的训练数据集，并对其预处理
mnist_data = torchvision.datasets.MNIST("./mnist", train=True, download=True, transform=transform)
# 将预处理好的数据集处理成dataloader
dataloader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=batch_size, shuffle=True)

# 初始化模型
D = D(image_size, hidden_size, num_classes).to(device)
G = G(latent_size, hidden_size, image_size, num_classes).to(device)
loss_func = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

# 训练循环
for epoch in range(epochs):
    for i, (images, labels) in enumerate(dataloader):
        batch_size = images.size(0)
        real_images = images.view(batch_size, image_size).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # 训练判别器
        y_preds = D(real_images, labels.to(device))
        d_loss_real = loss_func(y_preds, real_labels)
        
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z, labels.to(device))

        y_preds = D(fake_images.detach(), labels.to(device))
        d_loss_fake = loss_func(y_preds, fake_labels)
        
        # 更新判别器
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # 训练生成器
        z = torch.randn(batch_size, latent_size).to(device)  # 生成新的随机噪声
        fake_images = G(z, labels.to(device))
        y_preds = D(fake_images, labels.to(device))
        g_loss = loss_func(y_preds, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
    if epoch % 1 == 0:
        print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

    if epoch % 10 == 0:  # 每10个epoch保存一次
        save_generated_images(epoch)

# 保存模型
torch.save(D.state_dict(), 'discriminator.pth')
torch.save(G.state_dict(), 'generator.pth')

# 用生成器生成一张指定数字的图片并显示
def generate_digit_image(generator, digit):
    z = torch.randn(1, latent_size).to(device)
    label = torch.tensor([digit]).to(device)
    fake_image = generator(z, label).view(28, 28).data.cpu().numpy()
    plt.imshow(fake_image, cmap='gray')
    plt.axis('off')
    plt.show()

# 生成数字 3 的图像
generate_digit_image(G, 3)
