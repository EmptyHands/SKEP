import os
import copy
import argparse
import numpy as np
from functools import partial
import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.metrics import AccuracyAndF1
from paddlenlp.transformers import SkepTokenizer, SkepModel, LinearDecayWithWarmup
from utils.utils import set_seed
from utils.data import convert_example_to_feature

train_ds, dev_ds = load_dataset("chnsenticorp", splits=["train", "dev"])


model_name = "skep_ernie_1.0_large_ch"
batch_size = 16
max_seq_len = 256

tokenizer = SkepTokenizer.from_pretrained(model_name)
trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, max_seq_len=max_seq_len)
train_ds = train_ds.map(trans_func, lazy=False)
dev_ds = dev_ds.map(trans_func, lazy=False)

batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="int64")
    ): fn(samples)

train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=batch_size, shuffle=True)
dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=batch_size, shuffle=False)
train_loader = paddle.io.DataLoader(train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn)
dev_loader = paddle.io.DataLoader(dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn)

class SkepForSquenceClassification(paddle.nn.Layer):
    def __init__(self, skep, num_classes=2, dropout=None):
        super(SkepForSquenceClassification, self).__init__()
        self.num_classes = num_classes
        self.skep = skep
        self.dropout = paddle.nn.Dropout(p=dropout if dropout is not None else self.skep.config["hidden_dropout_prob"])
        self.classifier = paddle.nn.Linear(self.skep.config["hidden_size"], self.num_classes)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, pooled_output = self.skep(input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

# model hyperparameter  setting
num_epoch = 1
learning_rate = 3e-5
weight_decay = 0.01
warmup_proportion = 0.1
max_grad_norm = 1.0
log_step = 50
eval_step = 200
seed = 1000
checkpoint = "./checkpoint/"

set_seed(seed)
use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device("gpu:0")
if not os.path.exists(checkpoint):
    os.mkdir(checkpoint)

skep = SkepModel.from_pretrained(model_name)
model = SkepForSquenceClassification(skep, num_classes=len(train_ds.label_list))

num_training_steps = len(train_loader) * num_epoch
lr_scheduler = LinearDecayWithWarmup(learning_rate=learning_rate, total_steps=num_training_steps, warmup=warmup_proportion)
decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)
optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=weight_decay, apply_decay_param_fun=lambda x: x in decay_params, grad_clip=grad_clip)

metric = AccuracyAndF1()

def evaluate(model, data_loader, metric):

    model.eval()
    metric.reset()
    for idx, batch_data in enumerate(data_loader):
        input_ids, token_type_ids, labels = batch_data
        logits = model(input_ids, token_type_ids=token_type_ids)        

        # count metric
        correct = metric.compute(logits, labels)
        metric.update(correct)

    accuracy, precision, recall, f1, _ = metric.accumulate()

    return accuracy, precision, recall, f1

def train():
    global_step, best_f1 = 1, 0.
    model.train()
    for epoch in range(1, num_epoch+1):
        for batch_data in train_loader():
            input_ids, token_type_ids, labels = batch_data
            logits = model(input_ids, token_type_ids=token_type_ids)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % log_step == 0:
                print(f"epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}")
            if (global_step > 0 and global_step % eval_step == 0) or global_step == num_training_steps:
                accuracy, precision, recall, f1  = evaluate(model, dev_loader,  metric)
                model.train()
                if f1 > best_f1:
                    print(f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    paddle.save(model.state_dict(), f"{checkpoint}/best.pdparams")
                print(f'evalution result: accuracy: {accuracy:.5f}, precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')

            global_step += 1

    paddle.save(model.state_dict(), f"{checkpoint}/final.pdparams")

train()

# process data related
model_name = "skep_ernie_1.0_large_ch"
model_path = os.path.join(checkpoint, "best.pdparams")
id2label = {0:"Negative", 1:"Positive"}

# load model
loaded_state_dict = paddle.load(model_path)
ernie = SkepModel.from_pretrained(model_name)
model = SkepForSquenceClassification(ernie, num_classes=len(train_ds.label_list))    
model.load_dict(loaded_state_dict)

def predict(input_text, model, tokenizer, id2label, max_seq_len=256):

    model.eval()
    # processing input text
    encoded_inputs = tokenizer(input_text, max_seq_len=max_seq_len)
    input_ids = paddle.to_tensor([encoded_inputs["input_ids"]])
    token_type_ids = paddle.to_tensor([encoded_inputs["token_type_ids"]])

    # predict by model and decoding result 
    logits = model(input_ids, token_type_ids=token_type_ids)
    label_id =  paddle.argmax(logits, axis=1).numpy()[0]

    # print predict result
    print(f"text: {input_text} \nlabel: {id2label[label_id]}")

input_text = "交通方便；环境很好；服务态度很好 房间较大"
predict(input_text, model, tokenizer, id2label, max_seq_len=256)