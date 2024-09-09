import numpy as np
from datasets import Dataset, load_dataset
from dataclasses import dataclass, field

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Qwen2Tokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from transformers.trainer_callback import TrainerControl, TrainerState

TRAIN_FILES = [
    './datasets/wiki.parquet',
]

EVAL_FILE = "./datasets/pretrain_eval_512_1w.parquet"

# %%


@dataclass
class PretrainArguments:
    tokenizer_dir: str = "./model_save/"
    model_save_dir: str = "./model_save/pre/"
    logs_dir: str = "./logs/"
    train_files: list = field(default_factory=lambda: TRAIN_FILES)
    eval_file: str = EVAL_FILE
    max_seq_len: int = 512


pretrain_args = PretrainArguments()

tokenizer = Qwen2Tokenizer.from_pretrained(pretrain_args.tokenizer_dir)
vocab_size = len(tokenizer)
if vocab_size % 64 != 0:
    vocab_size = (vocab_size // 64 + 1) * 64

print(f"final vocab size: {vocab_size}")

map_dtype = np.uint16 if vocab_size < 65535 else np.uint32
def token_to_id(samples: dict) -> dict:
    batch_txt = samples["text"]
    outputs = tokenizer(
        batch_txt,
        padding=False,
        return_attention_mask=False,
        truncation=True,
        max_length=pretrain_args.max_seq_len
    )

    input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]

    return {"input_ids": input_ids}


def get_maped_dataset(files) -> Dataset:
    dataset = load_dataset(path="parquet", data_files=files, split="train", cache_dir=".cache", keep_in_memory=False)
    maped_dataset = dataset.map(token_to_id, batched=True, batch_size=10000, remove_columns=dataset.column_names, num_proc=1, keep_in_memory=False)
    return maped_dataset


train_dataset = get_maped_dataset(pretrain_args.train_files)
eval_dataset = get_maped_dataset(pretrain_args.eval_file)

# `mlm=False`表示要训练CLM模型，`mlm=True`表示要训练MLM模型
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

config = Qwen2Config.from_pretrained(pretrain_args.tokenizer_dir)
model = Qwen2ForCausalLM(config)

model_size = sum(t.numel() for t in model.parameters())
print(f"QWen size: {model_size / 1000**2:.1f}M parameters")


class MyTrainerCallback(TrainerCallback):
    log_cnt = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = True
        return control

my_trainer_callback = MyTrainerCallback()
args = TrainingArguments(
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    ddp_find_unused_parameters=False,
    warmup_steps=0,
    learning_rate=1e-4,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=50,
    save_strategy="steps",
    save_total_limit=4,
    report_to="tensorboard",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=20,
    log_level="info",
    logging_first_step=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[my_trainer_callback],
)

# `resume_from_checkpoint=True`参数可以从上次保存的检查点继续训练
trainer.train(  #'model_save/pre/checkpoint-3400'
    # resume_from_checkpoint=True
)

eval_results = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
trainer.save_model(pretrain_args.model_save_dir)
