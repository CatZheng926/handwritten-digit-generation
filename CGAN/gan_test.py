import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_size = 128
num_classes = 10  # 0-9

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

# 加载模型
def load_models():
    G_model = G(latent_size, 256, 28 * 28, num_classes).to(device)
    D_model = D(28 * 28, 256, num_classes).to(device)
    
    G_model.load_state_dict(torch.load('generator.pth', map_location=device))
    D_model.load_state_dict(torch.load('discriminator.pth', map_location=device))
    
    return D_model, G_model

# 生成数字图像
def generate_digit_image(generator, digit):
    z = torch.randn(1, latent_size).to(device)
    label = torch.tensor([digit]).to(device)
    fake_image = generator(z, label).view(28, 28).data.cpu().numpy()
    plt.imshow(fake_image, cmap='gray')
    plt.axis('off')
    plt.title(f'Generated Digit: {digit}')
    plt.show()
    return fake_image

# 评估生成图像
def evaluate_generated_image(discriminator, image):
    image = torch.tensor(image).view(1, -1).to(device)
    with torch.no_grad():
        probability = discriminator(image, torch.tensor([0]).to(device))  # 用任意标签进行评估
    return probability.item()

# 主函数
if __name__ == '__main__':
    D, G = load_models()  # 加载训练好的模型

    # 生成和评估特定数字
    digit_to_generate = 1  # 你想生成的数字
    fake_image = generate_digit_image(G, digit_to_generate)
    probability = evaluate_generated_image(D, fake_image)
    print(f"Discriminator Probability for generated digit '{digit_to_generate}': {probability:.4f}")
