import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

print("程序启动")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("./model_save/pre", device_map=device)
tokenizer = AutoTokenizer.from_pretrained("./model_save/pre")

while True:
    print("我：", end="")
    text = input()
    messages = [{"role": "user", "content": text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)