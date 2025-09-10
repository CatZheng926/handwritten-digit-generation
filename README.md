# handwritten-digit-generation
A **Conditional VAE** and a **GAN** implemented in **PyTorch** for generating handwritten digits from the **MNIST** dataset.
-----

这是一个利用深度生成模型生成手写数字的 Python 项目。本项目实现了**变分自编码器（VAE）和生成对抗网络（GAN）**，旨在比较和探索两种模型在图像生成任务中的表现。

-----

### ✨ 主要特性

  * **双模型实现**：包含一个条件变分自编码器（CVAE）和一个条件生成对抗网络（CGAN）。
  * **黑箱模型对比**：代码结构清晰，方便你对比两种模型在生成质量、训练稳定性和潜在空间特性上的差异。
  * **条件生成**：你可以根据指定的数字标签生成对应的手写数字图片。
  * **潜在空间探索**：通过操纵 VAE 的潜在空间，可以实现平滑的图像插值和风格转换。

### 🚀 快速开始

#### 1\. 克隆仓库

打开你的终端或 Git Bash，运行以下命令：

```bash
git clone https://github.com/CatZheng926/handwritten-digit-generation.git
cd handwritten-digit-generation
```

#### 2\. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3\. 运行模型

  * **训练 测试 VAE 模型**：

    ```bash
    python vae_train.py
    python vae_test.py
    ```

  * **训练 测试 GAN 模型**：

    ```bash
    python cgan_train.py
    python gan_test.py
    ```

### 🖼️ 结果展示

#### VAE 生成样本



#### GAN 生成样本
训练0，40，80，120次

<img width="800" height="800" alt="generated_epoch_0" src="https://github.com/user-attachments/assets/56ecb48b-c068-4b4e-b5e0-5bf1d63d4d02" />
<img width="800" height="800" alt="generated_epoch_40" src="https://github.com/user-attachments/assets/a4a0e7df-cb7b-42ec-9aee-e6016ea1fd61" />
<img width="800" height="800" alt="generated_epoch_80" src="https://github.com/user-attachments/assets/6bb7cdbc-bee9-4cb8-bc15-00c1db4d1700" />
<img width="800" height="800" alt="generated_epoch_120" src="https://github.com/user-attachments/assets/a28977c9-6200-4e50-a5bb-ebebd52b96e1" />


-----
