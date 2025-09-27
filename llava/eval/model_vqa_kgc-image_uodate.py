import sys

sys.path.append("/home/mw/work/mkgc/IT-MKGC")
import argparse
import logging

import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


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
    result = []
    for line in tqdm(questions):
        qs = line["conversations"][0]["value"]

        if "image" in line:
            image_file = line["image"]
            image_file = image_file.replace("\\", "/")
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0].half()  # torch.float32
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.split('\n')[1:]

        else:
            crop_size = image_processor.crop_size
            images = torch.zeros(3, crop_size['height'], crop_size['width']).unsqueeze(0).half().cuda()
            image_sizes = None

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        embedding_ids = torch.tensor(line["embedding_ids"]).unsqueeze(0).cuda()
        model = model.half()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                embedding_ids=embedding_ids,
                do_sample=True,  # 使用采样
                num_return_sequences=9,
                repetition_penalty=1.2,
                num_beams=1,  # 使用采样时设为1
                num_beam_groups=1,  # 使用采样时设为1
                temperature=1.2,
                top_p=0.8,
                no_repeat_ngram_size=2,
                max_new_tokens=30,
                use_cache=True
            )
        torch.cuda.empty_cache()

        outputs = [text.strip() for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
        result.append(
            {
                "answer": line["conversations"][1]["value"],
                "predict": outputs
            }
        )

    json.dump(result, ans_file, ensure_ascii=False, indent=4)


# 下载 NLTK 资源
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# 计算 Jaccard 相似度
def jaccard_similarity(str1, str2):
    words1 = set(nltk.word_tokenize(str1.lower()))
    words2 = set(nltk.word_tokenize(str2.lower()))
    return len(words1 & words2) / len(words1 | words2)


# 计算基于 WordNet 的单词相似度
def wordnet_similarity(word1, word2):
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    max_sim = 0
    for syn1 in synsets1:
        for syn2 in synsets2:
            sim = syn1.wup_similarity(syn2)  # Wu-Palmer 相似度
            if sim and sim > max_sim:
                max_sim = sim
    return max_sim if max_sim > 0.8 else 0  # 设定相似度阈值 0.8


# 计算基于 TF-IDF 的短语相似度（用于短语-短语）
def tfidf_similarity(str1, str2):
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]


# 计算单词或短语的匹配情况
def is_similar(answer, prediction):
    # 完全匹配
    if answer == prediction:
        return True

    # Jaccard 相似度（针对短语）
    if jaccard_similarity(answer, prediction) > 0.5:
        return True

    # 词级别匹配
    words1 = nltk.word_tokenize(answer.lower())
    words2 = nltk.word_tokenize(prediction.lower())

    match_count = 0
    for w1 in words1:
        for w2 in words2:
            sim = wordnet_similarity(w1, w2)
            if sim >= 0.7:  # 词义相似
                match_count += 1

    # 词汇层面匹配超过 50%，认为短语匹配
    if match_count / max(len(words1), len(words2)) >= 0.5:
        return True

    # TF-IDF 短语相似度
    tf_sim = tfidf_similarity(answer, prediction)
    if tf_sim > 0.7:
        return True

    return False


def eval_score(args):
    result = json.load(open(os.path.expanduser(args.answers_file), "r", encoding='utf-8'))
    hit_at_1, hit_at_3, hit_at_10, rank_sum = 0, 0, 0, 0
    total_samples = len(result)

    for res in result:
        true_answer = res["answer"]
        predictions = res["predict"]
        unique_predictions = list(dict.fromkeys(predictions))
        best_rank = float('inf')

        for i, pred in enumerate(unique_predictions):
            if is_similar(true_answer, pred):
                best_rank = min(best_rank, i + 1)

        if best_rank == 1:
            hit_at_1 += 1
        if best_rank <= 3:
            hit_at_3 += 1
        if best_rank <= 10:
            hit_at_10 += 1
        if best_rank != float('inf'):
            rank_sum += best_rank

    hit_at_1_score = hit_at_1 / total_samples
    hit_at_3_score = hit_at_3 / total_samples
    hit_at_10_score = hit_at_10 / total_samples
    mean_rank = rank_sum / total_samples

    print(f"Hit@1: {hit_at_1_score:.4f}")
    print(f"Hit@3: {hit_at_3_score:.4f}")
    print(f"Hit@10: {hit_at_10_score:.4f}")
    print(f"Mean Rank (MR): {mean_rank:.4f}")


'''
def eval_score(args):
    result = json.load(open(os.path.expanduser(args.answers_file), "r"))
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

        if true_answer in predictions:
            rank = predictions.index(true_answer) + 1  # 排名，+1表示从1开始
        else:
            rank = float('inf')

        # 更新指标
        if rank == 1:
            hit_at_1 += 1
        if rank <= 3:
            hit_at_3 += 1
        if rank <= 10:
            hit_at_10 += 1
        if rank != float('inf'):
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
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="../train/checkpoints/llava-v1.5-7b-task-lora-lr2e-5-fb15-237-condidation")
    parser.add_argument("--model-base", type=str, default="/home/mw/work/mkgc/llava-v1.5-7b")
    # parser.add_argument("--model-base", type=str, default="E:\\MMKG\\KGC\\llava-v1.5-7b")
    parser.add_argument("--image-folder", type=str, default="../../verbalizer")
    parser.add_argument("--question-file", type=str,
                        default="../../verbalizer/wn18-verbalizer-wo-neighberhood/test_verbalized_update.json")
    # parser.add_argument("--question-file", type=str, default="../../verbalizer/wn18-verbalizer_update/test.json")
    parser.add_argument("--answers-file", type=str,
                        default="../../verbalizer/fb15k-237-verbalizer-condidate/test_answer-condidate.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--temperature", type=float, default=1)
    # parser.add_argument("--top_p", type=float, default=0.8)
    # parser.add_argument("--num_beams", type=int, default=5)
    # parser.add_argument("--num_predictions", type=int, default=40)
    args = parser.parse_args()

    print("Loading modal:", args.model_path)
    print("Loading question:", args.question_file)

    # eval_model(args)
    eval_score(args)
    # os.system("/usr/bin/shutdown")
