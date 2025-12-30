# PyTorch 学习资源库 📚

这个文件夹包含了全面的 PyTorch 深度学习学习资源，涵盖从基础到高级的各种主题。按照难度和应用领域组织，适合从初学者到中级开发者的学习路径。

---

## 📁 文件夹结构总览

```
Pytorch/
├── Basics/                      # ⭐ 入门基础
├── CNN_architectures/           # 卷积神经网络架构
├── GANs/                        # 生成对抗网络
├── huggingface/                 # HuggingFace 自然语言处理
├── image_segmentation/          # 图像分割
├── more_advanced/               # 高级技术
├── object_detection/            # 目标检测
├── others/                      # 其他工具和设置
├── pytorch_lightning/           # PyTorch Lightning 框架
└── recommender_systems/         # 推荐系统
```

---

## 📖 详细内容说明

### 1️⃣ **Basics/** - PyTorch 基础入门 ⭐

这是最重要的入门文件夹，包含所有必须掌握的基础概念。

| 文件名 | 功能说明 |
|-------|---------|
| **pytorch_tensorbasics.py** | 张量基础操作（创建、索引、切片等） |
| **pytorch_simple_fullynet.py** | 简单的全连接神经网络实现 |
| **pytorch_simple_CNN.py** | 基础卷积神经网络（CNN）实现 |
| **pytorch_rnn_gru_lstm.py** | 循环神经网络（RNN、LSTM、GRU）讲解 |
| **pytorch_bidirectional_lstm.py** | 双向LSTM实现 |
| **pytorch_transforms.py** | 数据转换和预处理技巧 |
| **pytorch_tensorboard_.py** | TensorBoard 可视化工具使用 |
| **pytorch_loadsave.py** | 模型的保存和加载方法 |
| **pytorch_init_weights.py** | 权重初始化方法 |
| **pytorch_std_mean.py** | 标准化和均值计算 |
| **pytorch_lr_ratescheduler.py** | 学习率调度器（动态调整学习率） |
| **pytorch_mixed_precision_example.py** | 混合精度训练（加速训练，节省内存） |
| **pytorch_pretrain_finetune.py** | 预训练模型的微调方法 |
| **pytorch_progress_bar.py** | 进度条显示工具 |
| **lightning_simple_CNN.py** | 使用 PyTorch Lightning 实现简单CNN |

#### 子文件夹：

| 文件夹 | 说明 |
|-------|------|
| **custom_dataset/** | 自定义数据集加载器（CSV格式） |
| **custom_dataset_txt/** | 自定义数据集加载器（文本格式） |
| **albumentations_tutorial/** | 数据增强库 Albumentations 的使用教程 |
| **set_deterministic_behavior/** | 设置随机种子以确保结果可复现 |
| **Imbalanced_classes/** | 处理不平衡数据集的方法 |
| **dataset/** | 存放下载的 MNIST 数据集 |

---

### 2️⃣ **CNN_architectures/** - 卷积神经网络架构

实现和学习各种经典的CNN架构。

| 文件名 | 架构 | 说明 |
|-------|------|------|
| **lenet5_pytorch.py** | LeNet-5 | 最早的CNN，用于手写数字识别 |
| **pytorch_vgg_implementation.py** | VGG | 深层网络，使用小卷积核 |
| **pytorch_resnet.py** | ResNet | 残差网络，解决梯度消失问题 |
| **pytorch_efficientnet.py** | EfficientNet | 高效的网络，性能与速度的平衡 |
| **pytorch_inceptionet.py** | Inception | 使用多个卷积核尺寸的并行结构 |

**应用场景**：图像分类、特征提取等。

---

### 3️⃣ **GANs/** - 生成对抗网络（生成模型）

从简单到复杂的各种GAN实现，用于生成逼真的图像。

| 文件夹 | 说明 | 复杂度 |
|-------|------|--------|
| **1. SimpleGAN/** | 最简单的GAN实现（全连接层） | ⭐ 入门 |
| **2. DCGAN/** | 深度卷积GAN（用卷积层替代全连接） | ⭐⭐ 初级 |
| **3. WGAN/** | Wasserstein GAN（改进的损失函数） | ⭐⭐⭐ 中级 |
| **4. WGAN-GP/** | WGAN + 梯度惩罚（更稳定的训练） | ⭐⭐⭐ 中级 |
| **CycleGAN/** | 无配对图像翻译（如照片↔绘画） | ⭐⭐⭐⭐ 高级 |
| **Pix2Pix/** | 条件GAN，配对图像到图像翻译 | ⭐⭐⭐ 中级 |
| **SRGAN/** | 超分辨率GAN（低分辨率→高分辨率） | ⭐⭐⭐ 中级 |
| **ESRGAN/** | 增强型SRGAN（改进的超分辨率） | ⭐⭐⭐⭐ 高级 |
| **StyleGAN/** | 风格生成网络（高质量人脸生成） | ⭐⭐⭐⭐⭐ 专家级 |
| **ProGAN/** | 渐进式GAN（逐步增加分辨率） | ⭐⭐⭐⭐ 高级 |

**应用场景**：图像生成、图像翻译、超分辨率、数据增强等。

---

### 4️⃣ **image_segmentation/** - 图像分割

用于像素级的图像分析。

| 文件夹 | 说明 |
|-------|------|
| **semantic_segmentation_unet/** | U-Net 网络实现（医学图像分割等） |

**应用场景**：医学影像分析、自动驾驶、卫星图像处理等。

---

### 5️⃣ **object_detection/** - 目标检测

检测和定位图像中的物体。

| 文件夹 | 说明 |
|-------|------|
| **metrics/** | 目标检测评估指标（IoU、mAP等） |
| **YOLO/** | YOLO 检测算法 |
| **YOLOv3/** | YOLO v3 版本实现 |

**应用场景**：安全监控、人脸识别、自动驾驶等。

---

### 6️⃣ **more_advanced/** - 高级技术

各种高级和前沿的深度学习技术。

| 文件夹 | 说明 |
|-------|------|
| **Seq2Seq/** | 序列到序列模型（机器翻译基础） |
| **Seq2Seq_attention/** | 带注意力机制的Seq2Seq（改进版） |
| **seq2seq_transformer/** | 使用Transformer的Seq2Seq |
| **transformer_from_scratch/** | 从零实现Transformer模型 |
| **VAE/** | 变分自编码器（无监督学习） |
| **image_captioning/** | 图像标题生成（视觉+语言结合） |
| **neuralstyle/** | 神经风格迁移（艺术风格转换） |
| **torchtext/** | PyTorch 文本处理库教程 |
| **finetuning_whisper/** | Whisper 语音识别模型微调 |

**应用场景**：机器翻译、语音识别、图像理解、艺术生成等。

---

### 7️⃣ **huggingface/** - 自然语言处理（NLP）

使用 HuggingFace Transformers 库进行NLP任务。

| 文件名 | 功能 |
|-------|------|
| **learninghugg.py** | HuggingFace 基础学习 |
| **model.py** | 模型定义 |
| **train.py** | 模型训练脚本 |
| **test.py** | 模型测试脚本 |
| **dataset.py** | 数据集加载器 |
| **finetuning_t5_lightning.ipynb** | T5 模型微调（使用Lightning） |
| **finetune_t5_small_cnndaily.ipynb** | T5-small 在CNN/DailyMail数据集微调 |
| **cnndaily_t5_lightning_customdataloading.ipynb** | 自定义数据加载方式 |
| **learning.ipynb** | 学习笔记本 |

**应用场景**：文本分类、机器翻译、文本摘要、问答系统等。

---

### 8️⃣ **pytorch_lightning/** - PyTorch Lightning 框架

使用 PyTorch Lightning 简化模型训练（类似Keras对TensorFlow的作用）。

| 文件夹 | 说明 |
|-------|------|
| **1. start code/** | 开始使用 Lightning 的基础代码 |
| **2. LightningModule/** | Lightning 模块化组件 |
| **3. Lightning Trainer/** | 训练器配置和使用 |
| **4. Metrics/** | 评估指标计算 |
| **5. DataModule/** | 数据模块（数据处理管道） |
| **6. Restructuring/** | 代码重构和组织 |
| **7. Callbacks/** | 回调函数（早停、保存等） |
| **8. Logging Tensorboard/** | TensorBoard 日志记录 |
| **9. Profiler/** | 性能分析工具 |
| **10. Multi-GPU/** | 多GPU训练 |

**优势**：
- 代码简洁，专注模型逻辑
- 自动处理GPU/TPU
- 内置多GPU分布式训练
- 集成许多最佳实践

---

### 9️⃣ **recommender_systems/** - 推荐系统

构建推荐系统的模型和算法。

| 文件夹 | 说明 |
|-------|------|
| **neural_collaborative_filtering/** | 神经协同过滤（用户-物品交互建模） |

**应用场景**：电商推荐、视频推荐、音乐推荐等。

---

### 🔟 **others/** - 其他工具和设置

| 文件夹 | 说明 |
|-------|------|
| **default_setups/** | 默认配置和项目模板 |

---

## 🎯 学习路线建议

### 对于初学者：
1. **从Basics开始** - 按文件名顺序学习，理解PyTorch的基本概念
2. **pytorch_tensorbasics.py** → **pytorch_simple_fullynet.py** → **pytorch_simple_CNN.py**
3. 学习 **custom_dataset** 如何加载数据
4. 尝试 **pytorch_loadsave.py** 保存和加载模型

### 对于进阶学习者：
1. 学习 **CNN_architectures** 中的各种网络架构
2. 尝试 **GANs/1. SimpleGAN** 理解生成模型
3. 探索 **pytorch_lightning** 来优化代码结构

### 对于高级应用：
1. **GANs** 文件夹中的各种GAN变体
2. **more_advanced** 中的Transformer、VAE等
3. **huggingface** 进行自然语言处理任务
4. **object_detection** 和 **image_segmentation** 进行计算机视觉任务

---

## 🛠️ 环境和依赖

运行这些代码需要安装以下库：

```bash
# 核心库
pip install torch torchvision torchaudio

# 高级工具
pip install pytorch-lightning tensorboard transformers

# 数据处理
pip install numpy pandas scikit-learn albumentations

# 可视化
pip install matplotlib seaborn

# NLP工具
pip install torchtext huggingface-hub
```

---

## 💡 核心概念速查表

| 概念 | 所在位置 | 文件示例 |
|-----|--------|----------|
| **张量操作** | Basics | pytorch_tensorbasics.py |
| **神经网络构建** | Basics | pytorch_simple_fullynet.py |
| **CNN** | Basics, CNN_architectures | pytorch_simple_CNN.py, lenet5_pytorch.py |
| **RNN/LSTM** | Basics | pytorch_rnn_gru_lstm.py |
| **数据加载** | Basics | custom_dataset/custom_dataset.py |
| **模型保存/加载** | Basics | pytorch_loadsave.py |
| **生成模型** | GANs | 1. SimpleGAN/fc_gan.py |
| **迁移学习** | Basics | pytorch_pretrain_finetune.py |
| **Transformer** | more_advanced | transformer_from_scratch/ |
| **图像分割** | image_segmentation | semantic_segmentation_unet/ |
| **目标检测** | object_detection | YOLO/ |
| **框架简化** | pytorch_lightning | 1. start code/ |

---

## 📚 推荐学习资源

- **官方文档**：https://pytorch.org/docs/
- **PyTorch Tutorials**：https://pytorch.org/tutorials/
- **PyTorch Lightning**：https://www.pytorchlightning.ai/
- **HuggingFace Docs**：https://huggingface.co/docs/

---

## ✅ 快速开始

### 运行你的第一个PyTorch程序：

```bash
# 进入项目目录
cd d:\Machine-Learning-Collection\ML\Pytorch\Basics

# 运行基础教程
python pytorch_tensorbasics.py

# 或者运行一个简单的CNN
python pytorch_simple_CNN.py
```

---

**祝你学习愉快！🚀**

如有问题，欢迎查看各个文件的源代码注释，它们都有详细的说明。
