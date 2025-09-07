import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from modelscope import AutoModelForCausalLM as modelscope_AutoModelForCausalLM, \
    AutoTokenizer as modelscope_AutoTokenizer
from PIL import Image
import re
from qwen_vl_utils import process_vision_info

matplotlib.use('TkAgg', force=True)  # force=True 确保替换默认后端

data_path = "/mnt/nfs_fn/zsd_server/data/origin/mnist"

# 2. 明确指定split类型，直接返回Dataset对象而非DatasetDict
train_dataset = load_dataset(
    "parquet",
    data_files=os.path.join(data_path, "*/train*.parquet"),
    split="train",  # 关键优化：直接获取Dataset对象
    num_proc=4,  # 使用多进程加速加载（根据CPU核心数调整）
    keep_in_memory=False  # 大数据集时避免内存溢出
)
test_dataset = load_dataset(
    "parquet",
    data_files=os.path.join(data_path, "*/test*.parquet"),
    split="train",  # 关键优化：直接获取Dataset对象
    num_proc=4,
    keep_in_memory=False
)
print(train_dataset, test_dataset)

for i in range(min(2, len(train_dataset))):
    sample = train_dataset[i]
    print(f"\n样本 #{i + 1}: {sample}")
    # inverted_img = Image.eval(sample["image"], lambda x: 255 - x)
    # img_array = np.array(inverted_img)
    # plt.figure(figsize=(3, 3))  # 28x28图像适合小尺寸显示
    # plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')  # 隐藏坐标轴
    # plt.title("28x28 灰度图像")
    # plt.show()

# 加载模型和tokenizer
vl_model_path = "/mnt/nfs_fn/zsd_server/models/huggingface/Qwen2.5-VL-7B-Instruct"  # 根据实际模型路径调整
vl_tokenizer = AutoTokenizer.from_pretrained(vl_model_path, trust_remote_code=True)
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vl_model_path,
    device_map="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",
).eval()
processor = AutoProcessor.from_pretrained("/mnt/nfs_fn/zsd_server/models/huggingface/Qwen2.5-VL-7B-Instruct")


def analyze_image_with_qwen_vl(image, model, tokenizer, prompt_text):
    # 使用模型进行推理
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


    # 提取思考过程
    # think_pattern = r'<think>(.*?)</think>'
    # match = re.search(think_pattern, response, re.DOTALL)
    #
    # if match:
    #     return match.group(1).strip()
    # else:
    #     return "未能提取到思考过程"


prompt_text = "你是一个视觉语言模型。当用户提供一张图片时，请站在如何生成这张图片的角度，分析图片内容并输出一个思考过程。思考过程必须严格以<think>开始，以</think>结束。在思考过程中，详细描述图片中的对象、场景、细节以及你的推理。不要输出任何图像数据、token或其他文本；只输出思考部分。"
inverted_img = Image.eval(train_dataset[0]["image"], lambda x: 255 - x)
think_content = analyze_image_with_qwen_vl(image=inverted_img, model=vl_model, tokenizer=vl_tokenizer,
                                           prompt_text=prompt_text)
print(think_content)
"""
['<think>\n这张图片展示了一个简单的数字“5”。它是一个黑色的数字，背景是白色的。这个数字可能是用某种字体或书写工具绘制的。由于图片非常简洁，没有其他背景元素或装饰，因此无法提供更多的场景或细节信息。这个数字可能代表了某个计数、编号或是数学表达的一部分。\n</think>']
"""
