import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

print("程序启动")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    "./model_save/pre/checkpoint-300",
    torch_dtype="auto",
    device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./model_save/pre/checkpoint-300")

while True:
    print("我：", end="")
    prompt = input()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)