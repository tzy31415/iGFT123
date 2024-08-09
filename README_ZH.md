# Low Resource场景

## 初始数据采样

为了模拟low resource场景，我们随机从数据集中的训练集中采样500个作为大语言模型的初始训练数据，并且将其的格式转化成可以用于大语言模型的有监督微调，操作代码如下

```
python utils/sample_init_data.py --dataset dataset_name --num sample_num
```

其中dataset_name是采样的数据集，包括BEIR的数据集，比如MSMARCO、FiQA、NQ等，sample_num代表采样的数据

## 大语言模型优化

### 有监督微调

我们采用llama-factory的方式来优化我们的query generator，首先是使用上述的初始数据来进行有监督微调，在LLaMA-Factory/下使用以下脚本进行有监督微调

```
llamafactory-cli train config/SFT.yaml
```

其中SFT.yaml是微调时使用的配置文件，配置内容如下，可以在其中修改主要的参数，包括使用的大语言模型、微调的超参数等，其中dataset_name代表使用微调的数据集

```
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

stage: sft
do_train: true
finetuning_type: lora
lora_target: all

dataset: dataset_name
template: llama3
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

output_dir: saves/llama3-8b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
```

### 生成数据

在我们的框架中，大语言模型作为query generator backbone来生成pseudo query，我们通过以下LLaMA-Factory/下的脚本来使用优化好的大语言模型来生成query

```
llamafactory-cli chat config/inference.yaml
```

其中inference.yaml配置文件如下，其中的adapter_name_or_path代表经过优化后的适配器

```
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft
template: llama3
finetuning_type: lora
```

### 强化学习

在之后我们会根据过滤器等模块对pseudo query的质量进行打分，之后使用强化学习的方式迭代式的优化该大语言模型

首先通过以下代码训练强化学习中必要的奖励模型

```
llamafactory-cli chat config/reward.yaml
```

其中的reward.yaml配置文件如下，其中reward_dataset代表经过打分排序后的数据集

```
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

stage: rm
do_train: true
finetuning_type: lora
lora_target: all

dataset: reward_dataset
template: llama3
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/reward
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```


我们通过PPO算法结合之前reward model来强化学习

```
llamafactory-cli train config/ppo.yaml
```

其中ppo.yaml如下，其中reward_model即刚才训练的reward model，dataset_name即为之前训练用的数据集5

```
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
reward_model: saves/llama3-8b/lora/reward

stage: ppo
do_train: true
finetuning_type: lora
lora_target: all

dataset: dataset_name
template: llama3
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

output_dir: saves/llama3-8b/lora/ppo
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

max_new_tokens: 512
top_k: 0
top_p: 0.9
```


## 数据质量过滤模块

生成的pseudo query并不能保证自身质量的优越性，为了解决此问题，我们设计了以下模块来过滤生成的query的质量，包括从离散数据、密集数据以及主动学习的角度来过滤生成数据

### 离散数据质量过滤模块

我们首先使用基于BM25的离散数据过滤器对生成的数据进行过滤，具体做法是使用pseudo query从候选corpus集合中检索出正确的corpus，以此为标准判断该pseudo query的质量，其中candidate_num代表候选corpus集合的大小

```
bash Filter/sparse_filter.sh --module BM25 --candidate_num 500
```

### 密集数据质量过滤模块

密集数据质量过滤模块引入了密集检索器，旨在使用不同的视角对数据进行质量过滤，使用预训练的DPR、MonoT5等模块进行过滤，代码如下

```
bash Filter/dense_filter.sh --module MonoT --candidate_num 500
```

### 基于主动学习的质量过滤模块

基于主动学习的模块考虑了一个pseudo query对代训练的检索器的性能提升，在这里我们使用预测损失的方式来训练该主动学习检索器，首先使用预训练的方法得到该检索器

```
bash Filter/al_train.sh --dataset dataset_name --module module_name
```

之后使用预训练好的损失预测器去预测pseudo query可能带来的损失变化

```
bash Filter/al_predict.sh --module module_name 
```

## 验证ColBERT

之后就可以根据经过过滤的pseudo query去训练Co0lBERT，我们根据SPTAR的方法范式来训练我们ColBERT，代码如下，

首先是整理数据，将其转化成合适的格式，其中的dataset_name代表待转化格式的数据集

```

python zhiyuan/retriever/dpr/train/gen_data_for_colbert.py --dataset dataset_name --exp_name no_aug

```





之后训练ColBERT模型，其中cuda_num是使用的cuda的编号，dataset_name是待训练的数据集,exp_name是训练好的模型的文件位置，max_step的最大训练布署，save_per_step是保存的频率

```
bash zhiyuan/retriever/col_bert/train_colbert.sh -g cuda_num -d dataset_name -e exp_name -m max_step -s save_per_step -b batch_size

```

最后使用该ColBERT进行检索，验证模型效果，step代表使用步数的模型进行测试，par代表测试使用的块大小

bash zhiyuan/retriever/col_bert/test_colbert.sh -g cuda_num -d dataset_name -eexp_name -p par -c step
