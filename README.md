## LFW 人脸识别项目（PCA + SVM）

本项目使用 Python 和 scikit-learn 实现了一个基于 LFW（Labeled Faces in the Wild）数据集的人脸识别系统。主要方法包括主成分分析（PCA）进行降维，以及支持向量机（SVM）进行分类。

------

### 项目结构

- `LFW_2.ipynb`: 主代码文件，包含所有数据加载、处理和模型训练/评估过程。

------

### 快速开始

#### 1. 克隆项目并进入目录

```bash
git clone https://your-repo-url.git
cd your-repo
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 运行 Notebook

使用 Jupyter 打开并运行 `LFW_2.ipynb`：

```bash
jupyter notebook LFW_2.ipynb
```

------

###  使用的数据集

- 数据源：`sklearn.datasets.fetch_lfw_people`

------

###  模型结构

- **预处理**：
  - 图像灰度化
  - 标准化
- **降维**：
  - 使用 PCA（主成分保留率为 95%）
- **分类器**：
  - 使用 `SVC`（支持向量分类器），调参后使用 RBF 核
- **评估指标**：
  - 精度（accuracy）
  - 分类报告（precision, recall, f1-score）
  - 混淆矩阵



------

### 依赖列表（requirements.txt）

```txt
numpy
matplotlib
scikit-learn
seaborn
jupyter
```

