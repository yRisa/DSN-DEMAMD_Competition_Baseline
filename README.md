# 领域事件多因果关联挖掘比赛基线

## 运行步骤

### 一、安装环境

```bash
conda create -n DSN python=3.10
conda activate DSN
pip install transformers>=4.40.0
pip install flask requests peft deepspeed optimum accelerate
```

### 二、数据预处理

首先，将数据集解压后放入./data文件夹；

随后

```bash
python training_data_generation.py
```

### 三、训练

首先，将合适的模型放入./sft/model；

随后

```bash
cd sft
bash finetune.sh
```

### 四、推理

```bash
cd qwen_api
python apilora.py
```

结果存放在./result中

## 基线分数

Micro-average F1 Score: 0.5591205665580828

Precision: 0.5848662123313834

Recall: 0.5355459833591046
