import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = "/mnt/nfs_fn/zsd_server/models/huggingface/Qwen3-4B"
lora_path = "./save_model"
# lora_path = "./save_model/checkpoint-2814"
device = 'auto'

tokenizer = AutoTokenizer.from_pretrained(lora_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device, )
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model=model,
                                  model_id=lora_path,
                                  device_map=device,
                                  use_safetensors=True
                                  )
model = model.merge_and_unload()
for n, p in model.named_parameters():
    if p.device.type == "meta":
        print(f"{n} is on meta!")

label = 7
messages = [{
    "role": "user",
    "content": f"请生成一个28*28像素的手写数字{label}。",
}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=1.0,
    pad_token_id=tokenizer.eos_token_id,
)

# 解码完整响应
full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(full_response)

t = """
<|im_start|>user
请生成一个28*28像素的手写数字7。<|im_end|>
<|im_start|>assistant
<think>
用户需要我生成一个手写数字“7”。图像应为28x28像素的灰度图，背景为白色，数字为黑色。数字应具有手写字体特征，笔画可能有轻微的不规则感，整体居于图像中央。
</think>

好的，这是一个手写数字 7 的图像：<|vision_start|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX248|><|PX237|><|PX237|><|PX237|><|PX238|><|PX237|><|PX237|><|PX237|><|PX237|><|PX237|><|PX237|><|PX237|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX254|><|PX231|><|PX118|><|PX112|><|PX112|><|PX84|><|PX3|><|PX3|><|PX3|><|PX3|><|PX3|><|PX3|><|PX2|><|PX3|><|PX3|><|PX3|><|PX3|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX233|><|PX3|><|PX3|><|PX3|><|PX3|><|PX3|><|PX3|><|PX3|><|PX3|><|PX3|><|PX3|><|PX3|><|PX2|><|PX3|><|PX3|><|PX3|><|PX3|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX248|><|PX137|><|PX75|><|PX75|><|PX75|><|PX75|><|PX75|><|PX75|><|PX75|><|PX75|><|PX75|><|PX75|><|PX75|><|PX75|><|PX30|><|PX3|><|PX3|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX144|><|PX3|><|PX3|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX38|><|PX3|><|PX3|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX240|><|PX34|><|PX3|><|PX3|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX154|><|PX3|><|PX3|><|PX3|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX64|><|PX3|><|PX3|><|PX119|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX64|><|PX3|><|PX20|><|PX223|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX0|><|PX2|><|PX19|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX2|><|PX3|><|PX24|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX2|><|PX3|><|PX173|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX2|><|PX38|><|PX243|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX2|><|PX3|><|PX167|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX2|><|PX3|><|PX167|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX2|><|PX3|><|PX167|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|PX_line_start|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX2|><|PX3|><|PX167|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX255|><|PX_line_end|><|vision_end|><|im_end|>
"""


def extract_image_from_tokens(token_ids, tokenizer, img_size=None):
    """
    从 token ID 序列中提取并还原图像

    参数:
    token_ids (list): 包含图像 token 的 ID 序列
    tokenizer: 用于 token 转换的 tokenizer
    img_size (tuple): 可选，期望的图像尺寸 (宽度, 高度)。如果为 None，则自动确定

    返回:
    PIL.Image: 还原后的图像
    """
    # 获取特殊 token 的 ID
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    px_line_start_id = tokenizer.convert_tokens_to_ids("<|PX_line_start|>")
    px_line_end_id = tokenizer.convert_tokens_to_ids("<|PX_line_end|>")

    # 查找图像区域的起始和结束位置
    try:
        start_idx = token_ids.index(vision_start_id) + 1
        end_idx = token_ids.index(vision_end_id, start_idx)
        image_tokens = token_ids[start_idx:end_idx]
    except ValueError:
        print("未找到完整的图像 token 区域")
        return None

    # 处理图像 token
    pixels = []
    current_line = []
    in_image = False
    in_line = False

    for token_id in image_tokens:
        # 检查是否为行开始标记
        if token_id == px_line_start_id:
            in_line = True
            current_line = []
            continue

        # 检查是否为行结束标记
        if token_id == px_line_end_id:
            in_line = False
            if current_line:  # 确保行不为空
                pixels.append(current_line)
            continue

        # 如果在行内，处理像素 token
        if in_line:
            # 获取 token 文本
            token_text = tokenizer.convert_ids_to_tokens(token_id)

            # 检查是否为像素 token
            if token_text.startswith("<|PX") and token_text.endswith("|>"):
                try:
                    # 提取像素值
                    px_value = int(token_text[4:-2])
                    current_line.append(px_value)
                except ValueError:
                    # 如果无法提取像素值，跳过
                    continue

    # 如果没有提取到任何像素数据
    if not pixels:
        print("未提取到任何像素数据")
        return None

    # 确定图像尺寸
    if img_size:
        width, height = img_size
    else:
        # 自动确定尺寸：宽度为最长行的长度，高度为行数
        width = max(len(line) for line in pixels)
        height = len(pixels)
        print(f"自动确定图像尺寸: {width}x{height}")

    # 创建图像数组，用0（黑色）填充
    img_array = np.zeros((height, width), dtype=np.uint8)

    # 填充像素值
    for y, line in enumerate(pixels):
        if y >= height:
            break  # 超出指定高度
        for x, px_value in enumerate(line):
            if x >= width:
                break  # 超出指定宽度
            img_array[y, x] = px_value

    # 创建 PIL 图像
    image = Image.fromarray(img_array, mode='L')

    return image


def display_image_from_token_ids(token_ids, tokenizer, img_size=None):
    """
    从 token ID 序列中提取图像并显示

    参数:
    token_ids (list): 包含图像 token 的 ID 序列
    tokenizer: 用于 token 转换的 tokenizer
    img_size (tuple): 可选，期望的图像尺寸 (宽度, 高度)
    """
    # 提取图像
    image = extract_image_from_tokens(token_ids, tokenizer, img_size)

    if image is None:
        print("无法提取图像")
        return

    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title("从 Token 还原的图像")
    plt.show()

    return image


def display_image_from_text(text, tokenizer, img_size=None):
    """
    从模型输出文本中提取图像 token 并显示图像

    参数:
    text (str): 包含图像 token 的模型输出文本
    tokenizer: 用于 token 转换的 tokenizer
    img_size (tuple): 可选，期望的图像尺寸 (宽度, 高度)
    """
    # 将文本转换为 token ID
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # 显示图像
    return display_image_from_token_ids(token_ids, tokenizer, img_size)


# 显示图像（自动确定尺寸）
display_image_from_text(full_response, tokenizer)
