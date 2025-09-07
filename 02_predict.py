import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = "/mnt/nfs_fn/zsd_server/models/huggingface/Qwen3-4B"
lora_path = "./save_model"

# 新增特殊标记
SPECIAL_TOKENS = {
    **{f"[PX{i}]": f"[PX{i}]" for i in range(256)}  # 灰度像素标记
}

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'additional_special_tokens': list(SPECIAL_TOKENS.values())})
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda',)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model=model,
                                  model_id=lora_path,
                                  device_map='cuda',
                                  use_safetensors=True
                                  )
model = model.merge_and_unload()
for n, p in model.named_parameters():
    if p.device.type == "meta":
        print(f"{n} is on meta!")

label = 7
messages = [{
    "role": "user",
    "content": f"请画一个{label}",
}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
)

# 解码完整响应
full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(full_response)