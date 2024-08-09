from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import random
from transformers import Trainer, TrainingArguments
import  json
import csv
import torch



def calculate_similarity_score(model, tokenizer, query, document): # 这个放这占位置就行，之后计算分数采用DR那个包就行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_text = f"Query: {query} Document: {document}"    
    # input_text = remove_special_phrases(input_text) # 这一步删除特殊token等之后再操作
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(inputs, return_dict_in_generate=True, output_scores=True,output_logits=True)
    prob_list = outputs.logits[0].cpu().numpy().flatten().tolist()
    true_value = prob_list[1176]
    false_value = prob_list[6136]
    decoded_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    # return  (decoded_output, 0) if decoded_output == 'false' else (decoded_output, softmax_true_false(true_value, false_value))






def initialize_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model.to(device), tokenizer


def write_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def read_jsonl(filename):
    id2doc = {}
    with open(filename, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            id1 = item['_id']
            text = item['text']
            id2doc[id1] = text
    return id2doc

def read_tsv(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        qid2cid = []
        for row in tqdm(reader):
            if row[2] == 'score':
                continue
            qid2cid.append([row[0],row[1]])
    return qid2cid
            

class MonoT5Dataset(Dataset):
    def __init__(self, train_data, tokenizer, max_length=512):  # 输入的包括tokenizer
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        input_text, label = self.train_data[idx]
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            label,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # inputs['labels'] = labels.input_ids
        # return inputs

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels = labels.input_ids.squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# 加载预训练的 MonoT5 模型和分词器

if __name__ == "__main__":
    epoch = 3.0
    model_name = "/root/LLM/monot5-base-msmarco"  # Replace with your model's path or name
    model, tokenizer = initialize_model(model_name)

    queries_dict = read_jsonl('/root/NQ/nq-train/queries.jsonl')
    corpus_dict = read_jsonl('/root/NQ/nq-train/corpus.jsonl')
    qrels_dict = read_tsv('/root/NQ/nq-train/qrels/train.tsv')
    count = 0
    train_data = []

    for qd in tqdm(qrels_dict):
        count += 1
        if count >= 5001:
            break
        q_id, c_id = qd
        query = queries_dict[q_id]
        document = corpus_dict[c_id]
        positive_text = f"Query: {query} Document: {document}"
        negative_id = random.choice(list(corpus_dict.keys()))
        while negative_id == c_id:
            negative_id = random.choice(list(corpus_dict.keys()))
        no_document = corpus_dict[negative_id]
        negative_text = f"Query: {query} Document: {no_document}"
        train_data.append([positive_text, 'true'])
        train_data.append([negative_text, 'false'])
    write_json(train_data,'/root/NQ/nq-train/train_data_5k.json')
    train_data = json.loads(open('/root/NQ/nq-train/train_data_5k.json', 'r').read())

    dataset = MonoT5Dataset(train_data, tokenizer)
    training_args = TrainingArguments(
        output_dir='./results',            # 输出结果目录
        num_train_epochs=epoch,                # 训练的轮数
        per_device_train_batch_size=8,     # 训练时每个设备的批量大小
        per_device_eval_batch_size=2,      # 验证时每个设备的批量大小
        warmup_steps=500,                  # 学习率预热步骤
        weight_decay=5e-6,                 # 权重衰减
        logging_dir='./logs',              # 日志目录
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(f'./new_data_fine-tuned-monot5-base-{epoch}')
    tokenizer.save_pretrained(f'./new_data_fine-tuned-monot5-base-{epoch}')
