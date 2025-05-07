 ML project

# 准备工作：

1. 选题方向：人脸识别
2. 实现方法：SVM，CNN
3. 准备工作：
   - SVM的实现方法我是在skit-learn 官网上（[使用特征脸和SVM的人脸识别示例 — scikit-learn 1.6.0 文档 - scikit-learn 机器学习库](https://scikit-learn.cn/1.6/auto_examples/applications/plot_face_recognition.html?utm_source=chatgpt.com)）找到的。
   - CNN的实现方法可以参考github开源项目：[cyz020403/LFW-CNN-Examples: LFW dataset recognition and matching tasks examples.](https://github.com/cyz020403/LFW-CNN-Examples?utm_source=chatgpt.com)
   - CNN+SVM的实现方式可以参考博客：[CNN+SVM模型实现图形多分类任务（SVM替换softmax分类器）_面向图像识别的互补多分类器-CSDN博客](https://blog.csdn.net/weixin_40651515/article/details/105898211))



# write_up:

1. 可视化以及调参过程
2. 解决思路



## 一、proposal：

简单陈述project的实现过程和思路，这一步不需要很详细。但是为了能在课上和老师助教交流想法的时候能具体，还是早点开始动手做起来，根据得到的反馈才能发现问题进行内容扩充。



## 二、目前发现的问题

1. 优化SVM

   

1. ```py
   lfw_people = fetch_lfw_people(min_faces_per_person=?,resize=0.4)
   ```

   当选取的min_faces_per_person比较小，最后train的准确率非常低

分工（暂定）：

qys：SVM

ysz：调参和比较。

zhx：CNN

SVM：

lfw_people = fetch_lfw_people(*min_faces_per_person*=*5*,*resize*=*0.4*)

- **`min_faces_per_person=5`**
  数据集中包含许多只有 5~10 张样本的人物类别（如小众名人）。这些类别的样本过少，导致：
  
  - **训练不足**：SVM 需要足够的样本来学习每个类别的决策边界，样本过少时容易欠拟合。
  - **类别不平衡**：某些人物样本极少，模型会偏向样本多的类别，降低少数类的准确率。
  
- **`min_faces_per_person=70`**
  仅保留样本量≥70 的人物，此时：
  
  - 每个类别的样本更充足，SVM 能学到更稳定的分类边界。
  - 类别数量减少（如从 100+ 人减少到 10+ 人），降低了分类难度。
  
  我这里选取了比较折中的，代码里用的是50。

2. 一开始PCA提取到的特征感觉很模糊，边界不够清晰：

   解决：前面对样本进行标准化处理->

   延伸联想：还可以对data做哪些preprocess可以增强特征的提取？

3. **在使用了PCA提取特征之后在使用SVM，准确率有所提升。**

   **延伸：既然提取主要特征能提升准确率，那么可以考虑使用神经网络提升提取特征。再使用SVM。**

   同时可以考虑单纯使用CNN，最后一层用Softmax和最后一层用SVM来比较一下。

4. 由于单纯调整参数组合带来的收益太小了，准确率也不够高，所以采用另一种方法：cross-validation，10fold。

5. Cross-validation里面发现要标准化train 和testdata的数据最后结果能到96，否则的话，得分只有70左右。