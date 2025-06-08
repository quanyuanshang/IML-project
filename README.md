# LFW Face Recognition
---

## Method 1: PCA + SVM

<!-- 本部分使用 Python 和 scikit-learn 实现了一个基于 LFW（Labeled Faces in the Wild）数据集的人脸识别系统。主要方法包括增强鲁棒性的数据预处理（局部对比度增强CLAHE、边缘锐化增强、保边滤波），再通过主成分分析（PCA）进行降维，以及支持向量机（SVM）进行分类。 -->
This section implements a face recognition system based on the LFW (Labeled Faces in the Wild) dataset using Python and scikit-learn. The main approach includes robust data preprocessing (CLAHE for local contrast enhancement, edge sharpening, edge-preserving filtering), dimensionality reduction via Principal Component Analysis (PCA), and classification using a Support Vector Machine (SVM).

---
### Code Structure

- `LFW_2.ipynb`: The main code file, containing all data loading, processing, and model training/evaluation procedures.

---

<!-- ### 模型结构

- **预处理**：
  - 图像灰度化
  - 标准化
  - 数据增强
- **降维**：
  - 使用 PCA（主成分保留率为 95%）
- **分类器**：
  - 使用 `SVC`（支持向量分类器），调参后使用 RBF 核
- **评估指标**：
  - 精度（accuracy）
  - 分类报告（precision, recall, f1-score）
  - 混淆矩阵 -->
### Architecture
- **Preprocessing**:

  -Image grayscale conversion

  -Standardization

  -Data augmentation

- **Dimensionality Reduction**:

  -PCA (retaining 95% of principal components)

- **Classifier**:

  -SVC (Support Vector Classifier) with an RBF kernel after hyperparameter tuning

- **Evaluation Metrics**:

  -Accuracy

  -Classification report (precision, recall, f1-score)

  -Confusion matrix

---

## Method 2: Deep Learning Method
### Suite 1: PCA + Neural Network
* PCA-AlexNet
* PCA-ResNet18
* SmallCNN
* MLP

### Suite 2: End-to-End Neural Network
* AlexNet-Variant
* ResNet18-Variant
* CustomCNN
* Raw-Pixel MLP

---

## Quick Start

### 1. Clone the Project and Navigate to the Directory

```bash
git clone https://github.com/quanyuanshang/IML-project.git
cd IML-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.  Run the Notebook

* To reproduce the PCA+SVM method, open and run `LFW_2.ipynb` using Jupyter:

```bash
jupyter notebook LFW_2.ipynb
```
* To reproduce the PCA + Neural Network method, open and run `LFW_CNN.ipynb` using Jupyter:

```bash
jupyter notebook LFW_CNN.ipynb
```
* To reproduce the End-to-End Neural Network method, open and run `LFW_CNN_no_PCA.ipynb` using Jupyter:

```bash
jupyter notebook LFW_CNN_no_PCA.ipynb
```

---

### Dataset

- Data source:`sklearn.datasets.fetch_lfw_people`
---
### Dependency List (requirements.txt)

```txt
numpy
matplotlib
scikit-learn
seaborn
jupyter
torch
opencv-python
scikit-optimize
pandas
scipy
tqdm
torch
```
