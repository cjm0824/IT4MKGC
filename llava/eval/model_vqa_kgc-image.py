import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # 加载预训练模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
        questions = json.load(f)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    count = 0
    result = []
    for line in tqdm(questions):
        idx = line["id"]
        # image_file = line["image"]
        # qs = line["text"]
        # cur_prompt = qs
        qs = line["conversations"][0]["value"]
        cur_prompt = qs

        if "image" in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0].half()  # torch.float32
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.split('\n')[1:]
            # else:
            #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            images = None
            image_sizes = None

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        embedding_ids = torch.tensor(line["embedding_ids"]).unsqueeze(0).cuda()
        model=model.half()
        with torch.inference_mode():
            output_ids = model.generate(#生成模型的输出
                input_ids,
                images=images,#torch.float16
                image_sizes=image_sizes,
                embedding_ids=embedding_ids,
                do_sample=True if args.temperature > 0 else False,
                num_return_sequences= args.num_predictions,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=20,
                use_cache=True)
            # print(type(output_ids))

        outputs = [text.strip() for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
        result.append(
            {
                "answer": line["conversations"][1]["value"],
                "predict": outputs
            }
        )

    json.dump(result, ans_file, ensure_ascii=False, indent=4)

    # 初始化指标
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_10 = 0
    rank_sum = 0  # 用于计算MR
    total_samples = len(result)  # 样本总数

    for res in result:
        true_answer = res["answer"]  # 正确答案
        predictions = res["predict"]  # 模型预测结果，假设是一个列表
        predictions = [pred.strip() for pred in predictions]  # 去掉空格

        # 找到正确答案在预测结果中的排名
        try:
            rank = predictions.index(true_answer) + 1  # 排名，+1表示从1开始
        except ValueError:
            rank = float('inf')  # 如果正确答案不在预测结果中，排名为无穷大

        # 更新指标
        if rank == 1:
            hit_at_1 += 1
        if rank <= 3:
            hit_at_3 += 1
        if rank <= 10:
            hit_at_10 += 1
        rank_sum += rank  # 累加排名

    # 计算最终指标
    hit_at_1_score = hit_at_1 / total_samples
    hit_at_3_score = hit_at_3 / total_samples
    hit_at_10_score = hit_at_10 / total_samples
    mean_rank = rank_sum / total_samples

    # 打印结果
    print(f"Hit@1: {hit_at_1_score:.4f}")
    print(f"Hit@3: {hit_at_3_score:.4f}")
    print(f"Hit@10: {hit_at_10_score:.4f}")
    print(f"Mean Rank (MR): {mean_rank:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../train/checkpoints/llava-v1.5-7b-task-lora-lr2e-5-WN18")
    # parser.add_argument("--model-base", type=str, default="/root/autodl-tmp/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default="E:\\MMKG\\KGC\\llava-v1.5-7b")
    parser.add_argument("--image-folder", type=str, default="../../verbalizer")
    parser.add_argument("--question-file", type=str, default="../../verbalizer/wn18-verbalizer_update/test_verbalized_update.json")
    parser.add_argument("--answers-file", type=str, default="../../verbalizer/wn18-verbalizer_update/test_answer.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_predictions", type=int, default=20)
    args = parser.parse_args()

    eval_model(args)
    os.system("/usr/bin/shutdown")
