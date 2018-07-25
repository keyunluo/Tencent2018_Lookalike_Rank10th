## 代码
2018年腾讯广告算法大赛Rank10代码：深度部分。

## 环境说明

- 系统环境：　
    - 操作系统：Ubuntu16.04 LTS
    - 硬件：CPU　40核，128GB内存
    - 显卡: TITAN Xp, 显存：12G

- 软件环境：
    - Python: 3.6.4(conda 4.5.4)
    - Tensorflow: GPU版本１.7(源码编译),  NVIDIA-SMI 390.48, cuda_9.1.85_387.26
    - 其他包: pandas(0.22.0), numpy(1.14.0), scipy(1.0.0), scikit-leartn(0.19.1), tqdm(4.23.3)

## 运行步骤
- 预处理
    - 载入userFeature数据：`python3 load_vowpal.py `
- 生成特征
    - 生成uid/aid全局统计特征： `python3 make_feature.py`
- 运行模型
    - 运行NFFM模型: `python3 train.py`

## 特征工程

### 特征使用
- 基础特征：user、ad的所有基本特征
- 统计特征：uid_aid_nunique(每个uid下的aid数目，等频离散化), aid_uid_nunique(每个aid下的uid数目，等频离散化), campaignId_aid_nunique(每个campaignId下的aid数目， 等频离散化)，pos_aid(每个用户的训练集正aid)， neg_aid(每个用户的训练集负aid)，user_convert(用户转化率)

### 特征生成
- 全局统计特征：
    - 使用groupby提取，如`ad_Feature.groupby(['campaignId']).aid.nunique()`
    - 离散化：按百分比：[0, 20, 35, 50, 65, 85, 100]进行离散化
    - 转化率特征： 用户在训练集上的转化率，为防止过拟合，在移除当前行的label基础上进行统计
- 正负aid特征：
    - 在训练集中构建uid:aid-label字典
    - 整理上述字典uid:[aid-pos, aid-neg], 其中aid-pos为label为1的aid列表， aid-neg为label为-1的aid列表
    - 构建测试集中的特征：直接将上述uid拼接到测试集的uid中
    - 构建训练集中的特征：将上述uid拼接到训练集的uid中，并移除aid-pos、aid-neg中每行出现的aid
    - 最终生成多值正负aid特征 

## 模型结构
模型使用郭达雅同学在群里的开源代码：`nffm-v3`, 即深度FFM模型。
### 结构定义
- FFM部分：线性部分+二阶隐式交叉
- 深度部分：两个隐藏层

### 参数
- 批大小：4096
- 迭代次数: 1
- 隐藏层： 256, 128
- 优化器： adam
- 学习率： 0.0002
- L2正则： 0.000002
- 嵌入大小： 8
- 随机种子：2018

## 感谢
感谢郭达雅大佬的[开源模型](https://github.com/guoday/Tencent2018_Lookalike_Rank7th)
