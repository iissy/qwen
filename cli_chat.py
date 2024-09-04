import torch
from transformers import GenerationConfig

from transformers import (
    Qwen2Tokenizer,
    Qwen2ForCausalLM,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen2ForCausalLM.from_pretrained("./model_save/pre/checkpoint-250").to(device)
tokenizer = Qwen2Tokenizer.from_pretrained("./model_save/pre/checkpoint-250")
gen_config = GenerationConfig(
    temperature=0.3,
    top_k=20,
    top_p=0.5,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=300,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

print("程序启动")

while True:
    print("我：", end="")
    text = input()

    text = "你是一个助手 用户: {text} 回答: ".format(text=text)
    tokend = tokenizer(text, add_special=False)
    input_ids, attention_mask = torch.LongTensor([tokend.input_ids]).to(
        device
    ), torch.LongTensor([tokend.attention_mask]).to(device)
    outputs = model.generate(
        inputs=input_ids, attention_mask=attention_mask, generation_config=gen_config
    )

    outs = tokenizer.decode(outputs[0].cpu().numpy())
    print("AI:" + outs.split("<|im_end|>")[0].split("回答: ")[1])