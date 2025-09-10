import os

import transformers

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from PIL import Image
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import datasets

data_path = "/mnt/nfs_fn/zsd_server/data/origin/mnist"
model_path = "/mnt/nfs_fn/zsd_server/models/huggingface/Qwen3-4B"
lora_path = "./save_model"

# 新增特殊标记
SPECIAL_TOKENS = {
    "<|PX_line_start|>": "<|PX_line_start|>",
    "<|PX_line_end|>": "<|PX_line_end|>",
    **{f"<|PX{i}|>": f"<|PX{i}|>" for i in range(256)}  # 灰度像素标记
}

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'additional_special_tokens': list(SPECIAL_TOKENS.values())})
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, )
model.resize_token_embeddings(len(tokenizer))
print(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"]
)
model = get_peft_model(model, lora_config)
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)


    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
model.print_trainable_parameters()
model.config.use_cache = False
model.gradient_checkpointing_enable()

train_dataset = load_dataset(
    "parquet",
    data_files=os.path.join(data_path, "*/train*.parquet"),
    split="train",  # 关键优化：直接获取Dataset对象
    num_proc=4,  # 使用多进程加速加载（根据CPU核心数调整）
    keep_in_memory=False  # 大数据集时避免内存溢出
)
print(train_dataset)
print(train_dataset[0])


def process_image(image, size=(28, 28)):
    img = image.convert('L').resize(size)
    pixels = np.array(img)
    return pixels


def image_to_token_sequence(pixels):
    """将像素值转换为token ID序列"""
    px_line_start_id = tokenizer.encode("<|PX_line_start|>", add_special_tokens=False)[0]
    px_line_end_id = tokenizer.encode("<|PX_line_end|>", add_special_tokens=False)[0]
    pixel_tokens = []
    for line_pixels in pixels:
        pixel_tokens.append(px_line_start_id)
        for p in line_pixels:
            pixel_tokens.append(tokenizer.convert_tokens_to_ids(f"<|PX{int(p)}|>"))
        pixel_tokens.append(px_line_end_id)
    return pixel_tokens


def mnist_prepare(batch):
    """处理批量数据"""
    tokenized_inputs = []
    tokenized_labels = []
    vision_start_id = tokenizer.encode("<|vision_start|>", add_special_tokens=False)[0]
    vision_end_id = tokenizer.encode("<|vision_end|>", add_special_tokens=False)[0]

    # 获取批量中的图像和标签
    images = batch["image"]
    labels = batch["label"]

    for i in range(len(images)):
        inverted_img = Image.eval(images[i], lambda x: 255 - x)
        label = labels[i]

        messages = [{
            "role": "user",
            "content": f"请生成一个28*28像素的手写数字{label}。",
        }]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        target_text = f"<think>\n用户需要我生成一个手写数字“{label}”。图像应为28x28像素的灰度图，背景为白色，数字为黑色。数字应具有手写字体特征，笔画可能有轻微的不规则感，整体居于图像中央。\n</think>\n\n好的，这是一个手写数字 {label} 的图像："

        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        image_tokens = image_to_token_sequence(process_image(image=inverted_img, size=(28, 28)))
        target_ids = target_ids + [vision_start_id] + image_tokens + [vision_end_id] + [tokenizer.eos_token_id]

        full_ids = input_ids + target_ids
        labels_ids = full_ids.copy()
        len_prompt = len(input_ids)
        labels_ids[:len_prompt] = [-100] * len_prompt

        tokenized_inputs.append(full_ids)
        tokenized_labels.append(labels_ids)

    return {
        "input_ids": tokenized_inputs,
        "labels": tokenized_labels
    }


data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer,
    # return_tensors="pt",
    # padding=True,
    # pad_to_multiple_of=1,
    # pad_to_multiple_of=ARGS.max_length,  # the max_length arg is unused to padding label
)

run_name = "line_token"
training_args = TrainingArguments(
    output_dir=lora_path,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    warmup_ratio=0.01,
    num_train_epochs=5,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    report_to="tensorboard",
    run_name=run_name,
    logging_dir=f"logging_dir/{run_name}",
)
# selected_dataset = train_dataset.select(range(1000))
# print("Selected dataset size:", len(selected_dataset))
selected_dataset = train_dataset
processed_dataset = selected_dataset.map(mnist_prepare, batched=True, remove_columns=train_dataset.column_names)
print("Processed dataset size:", len(processed_dataset))
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(
    lora_path,
    save_embedding_layers=True  # Ensures resized embeddings are saved
)
tokenizer.save_pretrained(lora_path)

# tensorboard --logdir=./logging_dir --host=0.0.0.0
