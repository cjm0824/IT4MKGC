import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
import pandas as pd
from loguru import logger
from pymongo import MongoClient
import time
import os
from os import listdir
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
import torch
import transformers
from llava import LlavaLlamaForCausalLM
import torch.nn.functional as F
import math
class Verbalizer:
    def __init__(self, base_dataset_collection, similarity_matrix=None, relation2index=None, entity2text=None, entity_description=None,
                 relation2text=None):

        self.base_dataset_collection = base_dataset_collection

        self.similarity_matrix = similarity_matrix
        self.relation2index = relation2index

        self.entity2text = entity2text
        self.relation2text = relation2text
        self.entity_description = entity_description

        self.sep = '[SEP]'

    # 从数据集中获取指定节点的邻域信息
    def json_get_neighborhood(self, node_id, relation_id=None, tail_id=None, limit=None):   #获取结点node_id的邻域信息
        neighs = []  # 存储邻域节点信息的列表

        # relation_id and tail_id excluded for test dataset
        # if tail_id == None:  # 如果tail_id为空，则获取head节点的邻域信息
        #     cursor = self.base_dataset_collection.find({'head': node_id}, {'_id': False})  # 查找所有以node_id为头结点的邻域信息
        #     cursor = cursor.limit(limit) if limit else cursor  # limit参数用来限制返回的结果数量
        #
        #     for doc in cursor:  # 将查询结果添加到neighs列表中
        #         neighs.append(doc)
        #
        #     cursor = self.base_dataset_collection.find({'tail': node_id}, {'_id': False})  # 查询以node_id为尾结点的邻域信息
        #     cursor = cursor.limit(limit) if limit else cursor
        #     for doc in cursor:
        #         doc['relation'] = doc['relation']  # 修改relation为反向关系inverse of relation
        #         neighs.append(doc)  # 将关系添加到neighs列表中

        # relation_id and tail_id included for train dataset verbalization to hide the target node
        # else:  # 如果tail_id不为空，则获取tail节点的邻域信息
        # 查询所有以node_id为头节点的邻域信息，但尾结点不等于tail_id或关系不等于relation_id
        cursor = self.base_dataset_collection.find(
            {"$or": [{'tail': {'$ne': tail_id}}, {'relation': {'$ne': relation_id}}],
             'head': node_id}, {'_id': False})
        cursor = cursor.limit(limit) if limit else cursor
        for doc in cursor:
            neighs.append(doc)
        # 查询所有以node_id为尾结点的邻域信息，但头结点不等于tail_id或关系不等于relation_id
        cursor = self.base_dataset_collection.find(
            {"$or": [{'head': {'$ne': tail_id}}, {'relation': {'$ne': relation_id}}],
             'tail': node_id}, {'_id': False})
        cursor = cursor.limit(limit) if limit else cursor
        for doc in cursor:
            neighs.append(doc)

        return neighs

    # 根据邻域三元组中的关系与查询三元组中的关系的语义相似度，对邻域三元组进行排序，选择最相关的邻域作为输入的上下文信息
    def json_verbalize(self, head, relation, tail=None, inverse=False): #将邻域三元组转化为 verbalization

        limit = 10

        if inverse:
            neighborhood = self.json_get_neighborhood(tail, relation, head, limit)

            relation = self.relation2text[relation]
            # 排序
            neighborhood.sort(key=lambda x:
            (self.similarity_matrix[self.relation2index[self.relation2text[x['relation']]]][
                self.relation2index[relation]]),
                              reverse=True)
            # 限制邻域大小
            neighborhood = neighborhood[:512]

            neighborhood_info = ""
            if neighborhood:  # 判断neighborhood列表是否有元素
                neighborhood_info = "Following are some neighborhood triple about {}:\n".format(self.entity2text[tail])

            verbalization = (
                "Here is a triple with head entity h unknown: (h, {}, {}), you need to determine an appropriate response to complete the triple.\nFollowing are the structural features of {}:\n<Structure>\nFollowing are some details about {}: \n{}. \n{}").format(
                relation, self.entity2text[tail], self.entity2text[tail],self.entity2text[tail], self.entity_description[tail], neighborhood_info
                )
        else:
            neighborhood = self.json_get_neighborhood(head, relation, tail, limit)

            relation = self.relation2text[relation]
            # 排序
            neighborhood.sort(key=lambda x:
            (self.similarity_matrix[self.relation2index[self.relation2text[x['relation']]]][
                self.relation2index[relation]]),
                              reverse=True)
            # 限制邻域大小
            neighborhood = neighborhood[:512]
            neighborhood_info = ""
            if neighborhood:  # 判断neighborhood列表是否有元素
                neighborhood_info = "Following are some neighborhood triple about {}:\n".format(self.entity2text[head])

            verbalization = (
                "Here is a triple with tail entity t unknown: ({}, {}, t), you need to determine an appropriate response to complete the triple.\nFollowing are the structural features of {}:\n<Structure>\nFollowing are some details about {}: \n{}\n{}").format(
                self.entity2text[head], relation, self.entity2text[head], self.entity2text[head], self.entity_description[head], neighborhood_info)

        verbalization += " ".join(
            list(map(lambda x: "(" + self.entity2text[x['head']] + ", " + self.relation2text[x['relation']] + ", " +
                               self.entity2text[x['tail']] + ")" + " {}".format(self.sep),
                     neighborhood)))
        return verbalization.strip()

def save_to_json(data, json_file_path):    #将数据保存为JSON格式

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

entity_img_path = "E:\\MMKG\\KGC\\fb15k_wn18_images\\FB15k-images"
entity_img_files = listdir(entity_img_path)

# 加载 CLIP 模型
# model = CLIPModel.from_pretrained("E:\\MMKG\\KGC\\clip-vit-large-patch14-336").to("cuda")
model_path = "E:\\MMKG\\KGC\\llava-v1.5-7b"
model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
processor = CLIPProcessor.from_pretrained("E:\\MMKG\\KGC\\clip-vit-large-patch14-336")
#提取图像特征
def get_model():
    """获取 LLaVA 模型"""
    return model

vision_tower = get_model().get_vision_tower()
vision_tower.load_model()
vision_tower.to("cuda", dtype=torch.bfloat16)
# 加载预训练模型的tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
def get_image_features(image_path):
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    if image.size[0] > 30 and image.size[1] > 30:
        image = processor(images=image, return_tensors="pt")['pixel_values'].to("cuda", dtype=torch.bfloat16)#[3,336,336]

        with torch.no_grad():  # 获取图像特征
            image_features = vision_tower(image).to(dtype=torch.bfloat16)
            projected_features = get_model().model.mm_projector(image_features).squeeze(0)
        return projected_features   #返回图像的特征向量 [576,4096]

def get_text_features(text):

    input_ids = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False
    ).input_ids.to("cuda")
    # tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    # print(tokens)
    with torch.no_grad():
        entity_features = get_model().model.embed_tokens(input_ids).squeeze(0)
    return entity_features
'''
# 根据相似度矩阵选择出相似度最高的7张图像
def select_top_similar_images(similarity_matrix, image_paths):
    avg_similarities = np.mean(similarity_matrix, axis=1)  # 计算每张图像与其他图像的平均相似度
    top_indices = np.argsort(avg_similarities)[::-1][:7]  # 选择平均相似度最高的7张图像

    return [image_paths[i] for i in top_indices]
'''
def select_top_similar_images(similarity_matrix, image_paths):
    # 使用 PyTorch 的 mean 方法计算每张图像与其他图像的平均相似度
    avg_similarities = similarity_matrix.mean(dim=1).to(dtype=torch.float32)  # 按行计算均值 [num_images]
    # top_indices = torch.argsort(avg_similarities, descending=True)[:7]  # 选择平均相似度最高的7张图像
    valid_indices = torch.where(avg_similarities >= 0.9)[0]

    return [image_paths[i] for i in valid_indices]


def get_top_similar_images(ent, ent_name):
    # 获取目录中的所有图像路径

    if ent in entity_img_files:
        ent_file = os.path.join(entity_img_path, ent)
        image_paths = [os.path.join(ent_file, file) for file in os.listdir(ent_file)]

        # 提取所有图像的特征，过滤掉返回 None 的图像
        image_features = []
        valid_image_paths = []

        for img in image_paths:
            features = get_image_features(img)
            if features is not None:  # 仅保留有效的图像特征和对应的图像路径
                image_features.append(features)
                valid_image_paths.append(img)
        ent_features = get_text_features(ent_name)
        global_text_features = torch.mean(ent_features, dim=0)  # 汇总文本特征

        # 剔除噪声图像
        threshold = 0.027
        valid_images = []
        similarity_ = []
        for i, img_features in enumerate(image_features):
            global_image_features = torch.mean(img_features, dim=0)  # 汇总图像特征
            similarity = F.cosine_similarity(global_text_features.unsqueeze(0), global_image_features.unsqueeze(0),
                                             dim=1)
            similarity_.append(similarity)
            if similarity.item() >= threshold:
                valid_images.append(valid_image_paths[i])  # 保留高相似度图像路径



        #     similarities.append(global_image_features.cpu().numpy())
        '''
        similarities = torch.stack([torch.mean(feature, dim=0) for feature in image_features],
                                   dim=0)  # [num_images, 4096]
        # 计算图像之间的相似度矩阵
        similarity_matrix = F.cosine_similarity(similarities.unsqueeze(1), similarities.unsqueeze(0), dim=2)  # [num_images, num_images]

        # 选择相似度最高的7张图像
        top_similar_images = select_top_similar_images(similarity_matrix, valid_image_paths)

        return top_similar_images
        '''
        return valid_images
'''
def concatenate_images_horizontal(images, dist_images):
    # calc total width of imgs + dist between them
    total_width = sum(img.width for img in images) + dist_images * (len(images) - 1)
    # calc max height from imgs
    height = max(img.height for img in images)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    # init var to track current width pos
    current_width = 0
    for img in images:
        # paste img in new_img at current width
        new_img.paste(img, (current_width, 0))
        # update current width for next img
        current_width += img.width + dist_images
    return new_img

def save_images_to_folder(image_files, folder_base_path, tail):
    """
    将图像保存到指定的文件夹中。

    :param image_files: 图像文件路径列表
    :param folder_base_path: 基础文件夹路径
    :param tail: 用于构建文件夹名称的标识
    """
    # 构建目标文件夹路径
    folder_path = os.path.join(folder_base_path, f"n{tail}")
    os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在则创建

    # 遍历图像文件列表并保存到目标文件夹
    for img_path in image_files:
        try:
            # 打开图像
            img = Image.open(img_path)
            if img.mode in ["P", "RGBA"]:
                img = img.convert("RGB")
            # 构造保存路径
            img_name = os.path.basename(img_path)  # 获取图像文件名
            save_path = os.path.join(folder_path, img_name)
            # 保存图像
            img.save(save_path)
        except Exception as e:
            print(f"Failed to save image {img_path}: {e}")
'''


def save_images_to_folder(image_files, folder_base_path, tail, grid_size=None, image_size=(224, 224), padding_color=(255, 255, 255)):
    """
    将图像以网格形式拼接为单张图像并保存到指定的文件夹中，支持处理大小不一致的图像。

    :param image_files: 图像文件路径列表
    :param folder_base_path: 基础文件夹路径
    :param tail: 用于构建文件夹名称的标识
    :param grid_size: 网格的行列数元组 (rows, cols)，若为 None 则自动计算为接近正方形
    :param image_size: 每张小图像的目标尺寸 (width, height)
    :param padding_color: 填充颜色，默认为白色
    """
    # 构建目标文件夹路径
    # folder_path = os.path.join(folder_base_path, f"n{tail}")
    # os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在则创建

    try:
        # 自动计算网格大小
        num_images = len(image_files)
        if grid_size is None:
            cols = math.ceil(math.sqrt(num_images))  # 列数
            rows = math.ceil(num_images / cols)  # 行数
        else:
            rows, cols = grid_size

        # 创建网格画布
        grid_width = cols * image_size[0]
        grid_height = rows * image_size[1]
        grid_image = Image.new("RGB", (grid_width, grid_height), padding_color)  # 填充背景颜色

        # 将图像逐一处理并放入网格
        for idx, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path)
                if img.mode in ["P", "RGBA"]:
                    img = img.convert("RGB")
                    # 动态选择填充颜色
                if img.mode in ["P", "L"]:  # P 模式或灰度模式
                    fill_color = 255  # 使用单个整数值
                else:  # RGB 模式
                        fill_color = padding_color  # 使用 RGB 元组

                # 按照目标大小填充图像以保持比例
                img = ImageOps.pad(img, image_size, color=fill_color)

                # 计算图像在网格中的位置
                row, col = divmod(idx, cols)
                x_offset = col * image_size[0]
                y_offset = row * image_size[1]

                # 粘贴到网格画布上
                grid_image.paste(img, (x_offset, y_offset))
            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")

        # 构造保存路径
        save_path = os.path.join(folder_base_path, f"{tail}-image.jpg")
        # 保存拼接后的图像
        grid_image.save(save_path)
        # print(f"Grid image saved at: {save_path}")

    except Exception as e:
        print(f"Failed to save grid image: {e}")
'''
def verbalize_dataset(input_df, output_collection, verbalizer, json_file_path=None):
    # logger.info('Started verbalizing {}th triplet'.format(i))
    start_time = time.time()
    json_docs = []
    for i, doc in tqdm(input_df.iterrows(), total=len(input_df)):
        try:
            #已知head，预测tail
            json_direct_verbalization = verbalizer.json_verbalize(doc['head'], doc['relation'], doc['tail'], inverse=False)
            # print("head:"+str(len(json_direct_verbalization.split(" "))))
            head = doc["head"][1:].replace('/', '.')
            head_name = verbalizer.entity2text[doc['head']]
            # head_path = "wn18-Images_update1/n" + head + "-image.jpg"
            # head_path = os.path.join("wn18-Images_update", f"n{head}")
            head_path = os.path.join("fb15k-237-Images_update", f"{head}-image.jpg")
            if os.path.exists(head_path):
                image_information = "Following are images associated with {} in the multimodal knowledge graph: \n<Image>\n".format(
                    verbalizer.entity2text[doc['head']])
                # files = os.listdir(head_path)
                # image_information = "Following are images associated with {} in the multimodal knowledge graph: \n".format(
                #     verbalizer.entity2text[doc['head']])
                # # 添加多个 <Image> 标签
                # image_information += "".join("<Image> " * len(files))
                # image_information = image_information.strip() + "\n"

                index = json_direct_verbalization.find("\n")
                value = json_direct_verbalization[:index + 1] + image_information + json_direct_verbalization[
                                                                                    index + 1:]
                json_docs.append({"id": i*2,
                                  "image": head_path,
                                  "embedding_ids": [entity_dict[doc['head']]],
                                  "conversations":[
                                      {
                                          "from": "human",
                                          "value": value
                                      },
                                      {
                                          "from": "gpt",
                                          "value": verbalizer.entity2text[doc['tail']]
                                     }
                                  ]
                })
            else:
                head_img_files = get_top_similar_images(head, head_name)
                if head_img_files is not None and len(head_img_files)>0:
                    save_images_to_folder(head_img_files, "fb15k-237-Images_update", head)
                    image_information = "Following are images associated with {} in the multimodal knowledge graph: \n<Image>\n".format(
                        verbalizer.entity2text[doc['head']])
                    # image_information = "Following are images associated with {} in the multimodal knowledge graph: \n".format(
                    #     verbalizer.entity2text[doc['head']])
                    # image_information += ''.join([f"<Image> " for _ in head_img_files])
                    # image_information += "\n"
                    index = json_direct_verbalization.find("\n")
                    value = json_direct_verbalization[:index+1]+image_information+json_direct_verbalization[index+1:]

                    json_docs.append({"id": i*2,
                                      "image": head_path,
                                      "embedding_ids": [entity_dict[doc['head']]],
                                      "conversations":[
                                      {
                                          "from": "human",
                                          "value": value
                                      },
                                      {
                                          "from": "gpt",
                                          "value": verbalizer.entity2text[doc['tail']]
                                     }
                                  ]
                    })
                else:
                    json_docs.append({"id": i * 2,
                                      "embedding_ids": [entity_dict[doc['head']]],
                                      "conversations": [
                                          {
                                              "from": "human",
                                              "value": json_direct_verbalization
                                          },
                                          {
                                              "from": "gpt",
                                              "value": verbalizer.entity2text[doc['tail']]
                                          }
                                      ]
                    })

            #已知tail，预测head
            json_direct_verbalization = verbalizer.json_verbalize(doc['head'], doc['relation'], doc['tail'], inverse=True)
            # print("tail:" + str(len(json_direct_verbalization.split(" "))))
            tail = doc["tail"][1:].replace('/', '.')
            tail_name = verbalizer.entity2text[doc['tail']]
            # tail_path = "wn18-Images_update1"# + tail + "-image.jpg"
            # tail_path = os.path.join("wn18-Images_update", f"n{tail}")
            tail_path = os.path.join("fb15k-237-Images_update", f"{tail}-image.jpg")
            if os.path.exists(tail_path):
                # files = os.listdir(tail_path)
                image_information = "Following are images associated with {} in the multimodal knowledge graph: \n<Image>\n".format(
                    verbalizer.entity2text[doc['tail']])
                # image_information = "Following are images associated with {} in the multimodal knowledge graph: \n".format(
                #     verbalizer.entity2text[doc['tail']])
                #
                # # 添加多个 <Image> 标签
                # image_information += "".join("<Image> " * len(files))
                # image_information = image_information.strip() + "\n"
                index = json_direct_verbalization.find("\n")
                value = json_direct_verbalization[:index + 1] + image_information + json_direct_verbalization[
                                                                                    index + 1:]
                json_docs.append({"id": i * 2 + 1,
                                  "image": tail_path,
                                  "embedding_ids": [entity_dict[doc['tail']]],
                                  "conversations": [
                                      {
                                          "from": "human",
                                          "value": value
                                      },
                                      {
                                          "from": "gpt",
                                          "value": verbalizer.entity2text[doc['head']]
                                      }
                                  ]
                })
            else:
                tail_img_files = get_top_similar_images(tail,tail_name)
                if tail_img_files is not None and len(tail_img_files)>0:
                    save_images_to_folder(tail_img_files, "fb15k-237-Images_update", tail)
                    # tail_imgs = [Image.open(img) for img in tail_img_files]
                    # tail_imgs = concatenate_images_horizontal(tail_imgs, 5)
                    # tail_imgs.save(tail_path)
                    image_information = "Following are images associated with {} in the multimodal knowledge graph: \n<Image>\n".format(
                        verbalizer.entity2text[doc['tail']])
                    # image_information += ''.join([f"<Image> " for _ in tail_img_files])
                    # image_information = image_information.strip() + "\n"
                    index = json_direct_verbalization.find("\n")
                    value = json_direct_verbalization[:index + 1] + image_information + json_direct_verbalization[
                                                                                        index + 1:]
                    json_docs.append({"id": i * 2 + 1,
                                      "image": tail_path,
                                      "embedding_ids": [entity_dict[doc['tail']]],
                                      "conversations": [
                                          {
                                              "from": "human",
                                              "value": value
                                          },
                                          {
                                              "from": "gpt",
                                              "value": verbalizer.entity2text[doc['head']]
                                          }
                                      ]
                    })
                else:
                    json_docs.append({"id": i * 2 + 1,
                                      "embedding_ids": [entity_dict[doc['tail']]],
                                      "conversations": [
                                          {
                                              "from": "human",
                                              "value": json_direct_verbalization
                                          },
                                          {
                                              "from": "gpt",
                                              "value": verbalizer.entity2text[doc['head']]
                                          }
                                      ]
                    })

        except Exception as e:
            logger.exception('Exception {} on {}th triplet'.format(e, i))
        if i % 100000 == 0:
            logger.info('verbalized {}th triple | spanned time: {}s'.format(i, int((time.time() - start_time))))
    if json_file_path:
        save_to_json(json_docs, json_file_path)
'''
from concurrent.futures import ThreadPoolExecutor
async def process_single_doc_async(doc, i, verbalizer, entity_dict, device, loop):
    """异步处理单个文档"""
    try:
        json_results = []

        # 创建一个线程池执行器来处理CPU密集型任务
        with ThreadPoolExecutor() as executor:
            # 处理 head prediction
            json_direct_verbalization = await loop.run_in_executor(
                executor,
                verbalizer.json_verbalize,
                doc['head'], doc['relation'], doc['tail'], False
            )

            head = doc["head"][1:].replace('/', '.')
            head_name = verbalizer.entity2text[doc['head']]
            head_path = os.path.join("fb15k-237-Images_update", f"{head}-image.jpg")

            if os.path.exists(head_path):
                image_information = "Following are images associated with {} in the multimodal knowledge graph: \n<Image>\n".format(
                    verbalizer.entity2text[doc['head']])
                index = json_direct_verbalization.find("\n")
                value = json_direct_verbalization[:index + 1] + image_information + json_direct_verbalization[
                                                                                    index + 1:]

                json_results.append({
                    "id": i * 2,
                    "image": head_path,
                    "embedding_ids": [entity_dict[doc['head']]],
                    "conversations": [
                        {"from": "human", "value": value},
                        {"from": "gpt", "value": verbalizer.entity2text[doc['tail']]}
                    ]
                })
            else:
                # 异步处理图像
                head_img_files = await process_images_async(head, head_name, device, loop)
                if head_img_files is not None and len(head_img_files) > 0:
                    await loop.run_in_executor(
                        executor,
                        save_images_to_folder,
                        head_img_files,
                        "fb15k-237-Images_update",
                        head
                    )

                    image_information = "Following are images associated with {} in the multimodal knowledge graph: \n<Image>\n".format(
                        verbalizer.entity2text[doc['head']])
                    index = json_direct_verbalization.find("\n")
                    value = json_direct_verbalization[:index + 1] + image_information + json_direct_verbalization[
                                                                                        index + 1:]

                    json_results.append({
                        "id": i * 2,
                        "image": head_path,
                        "embedding_ids": [entity_dict[doc['head']]],
                        "conversations": [
                            {"from": "human", "value": value},
                            {"from": "gpt", "value": verbalizer.entity2text[doc['tail']]}
                        ]
                    })
                else:
                    json_results.append({
                        "id": i * 2,
                        "embedding_ids": [entity_dict[doc['head']]],
                        "conversations": [
                            {"from": "human", "value": json_direct_verbalization},
                            {"from": "gpt", "value": verbalizer.entity2text[doc['tail']]}
                        ]
                    })

            # 处理 tail prediction
            json_direct_verbalization = await loop.run_in_executor(
                executor,
                verbalizer.json_verbalize,
                doc['head'], doc['relation'], doc['tail'], True
            )

            tail = doc["tail"][1:].replace('/', '.')
            tail_name = verbalizer.entity2text[doc['tail']]
            tail_path = os.path.join("fb15k-237-Images_update", f"{tail}-image.jpg")

            if os.path.exists(tail_path):
                image_information = "Following are images associated with {} in the multimodal knowledge graph: \n<Image>\n".format(
                    verbalizer.entity2text[doc['tail']])
                index = json_direct_verbalization.find("\n")
                value = json_direct_verbalization[:index + 1] + image_information + json_direct_verbalization[
                                                                                    index + 1:]

                json_results.append({
                    "id": i * 2 + 1,
                    "image": tail_path,
                    "embedding_ids": [entity_dict[doc['tail']]],
                    "conversations": [
                        {"from": "human", "value": value},
                        {"from": "gpt", "value": verbalizer.entity2text[doc['head']]}
                    ]
                })
            else:
                tail_img_files = await process_images_async(tail, tail_name, device, loop)
                if tail_img_files is not None and len(tail_img_files) > 0:
                    await loop.run_in_executor(
                        executor,
                        save_images_to_folder,
                        tail_img_files,
                        "fb15k-237-Images_update",
                        tail
                    )

                    image_information = "Following are images associated with {} in the multimodal knowledge graph: \n<Image>\n".format(
                        verbalizer.entity2text[doc['tail']])
                    index = json_direct_verbalization.find("\n")
                    value = json_direct_verbalization[:index + 1] + image_information + json_direct_verbalization[
                                                                                        index + 1:]

                    json_results.append({
                        "id": i * 2 + 1,
                        "image": tail_path,
                        "embedding_ids": [entity_dict[doc['tail']]],
                        "conversations": [
                            {"from": "human", "value": value},
                            {"from": "gpt", "value": verbalizer.entity2text[doc['head']]}
                        ]
                    })
                else:
                    json_results.append({
                        "id": i * 2 + 1,
                        "embedding_ids": [entity_dict[doc['tail']]],
                        "conversations": [
                            {"from": "human", "value": json_direct_verbalization},
                            {"from": "gpt", "value": verbalizer.entity2text[doc['head']]}
                        ]
                    })

        return json_results

    except Exception as e:
        logger.exception('Exception {} on {}th triplet'.format(e, i))
        return []

import asyncio
async def process_images_async(entity, entity_name, device, loop):
    """异步处理图像"""
    try:
        async with asyncio.Lock():  # 防止GPU内存冲突
            with torch.cuda.device(device):
                # 使用线程池执行器处理任务
                with ThreadPoolExecutor() as executor:
                    # 检查并转换输入数据类型
                    def process_with_gpu():
                        if isinstance(entity, torch.Tensor):
                            # 如果是张量，移到GPU
                            gpu_entity = entity.cuda(device)
                        elif isinstance(entity, np.ndarray):
                            # 如果是numpy数组，转换为张量后移到GPU
                            gpu_entity = torch.from_numpy(entity).cuda(device)
                        else:
                            # 如果是其他类型，直接使用原始数据
                            gpu_entity = entity

                        return get_top_similar_images(gpu_entity, entity_name)

                    # 将处理函数包装在 run_in_executor 中执行
                    result = await loop.run_in_executor(
                        executor,
                        process_with_gpu
                    )

                    # 如果需要在GPU上进行后续处理
                    if result is not None and len(result) > 0:
                        # 这里可以添加GPU相关的处理代码
                        pass

                    # 清理GPU内存
                    torch.cuda.empty_cache()
                    return result

    except Exception as e:
        logger.exception(f'Error processing images for {entity_name}: {e}')
        return None


async def process_batch_async(batch_data, verbalizer, entity_dict, device, loop):
    """异步处理批次数据"""
    tasks = []
    for doc, i in batch_data:
        task = asyncio.create_task(
            process_single_doc_async(doc, i, verbalizer, entity_dict, device, loop)
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]  # 展平结果列表


async def verbalize_dataset_async(input_df, verbalizer, json_file_path=None):
    """主异步处理函数"""
    start_time = time.time()

    # 设置GPU设备
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU devices found!")

    # 计算批次大小
    total_samples = len(input_df)
    batch_size = 10  # 根据GPU内存调整

    # 创建批次
    batches = []
    for i in range(0, total_samples, batch_size):
        batch_df = input_df.iloc[i:i + batch_size]
        batch_data = [(doc, idx) for idx, doc in batch_df.iterrows()]
        batches.append(batch_data)

    # 获取事件循环
    loop = asyncio.get_event_loop()

    # 处理批次
    json_docs = []
    with tqdm(total=len(batches)) as pbar:
        for batch_idx, batch in enumerate(batches):
            device = batch_idx % num_gpus
            batch_results = await process_batch_async(batch, verbalizer, entity_dict, device, loop)
            json_docs.extend(batch_results)
            pbar.update(1)

            # 定期清理GPU内存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # 保存中间结果
            if (batch_idx + 1) % 100 == 0:
                logger.info(f'Processed {batch_idx + 1} batches | Time elapsed: {int(time.time() - start_time)}s')

    if json_file_path:
        await loop.run_in_executor(
            None,
            lambda: save_json_file(json_docs, json_file_path)
        )

    logger.info('Total processing time: {}s'.format(int(time.time() - start_time)))

    return json_docs

def save_json_file(json_docs, json_file_path):
    """保存JSON文件的辅助函数"""
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_docs, f, ensure_ascii=False, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument("--relation_vectors_path", help="path to the embeddings of verbalized relations", default="fb15k-237-fasttext_relation_emb.npy")
parser.add_argument("--rel2ind_path", help="path to the mapping of textual relations to the index of corresponding vectors", default="../dataset/FB15k-237/relation2ind.json")
parser.add_argument("--entity_mapping_path", help="path to the entity2text mapping", default="../dataset/FB15k-237/entity2text.txt")
parser.add_argument("--entity_description_path", help="path to the entity2textlong mapping", default="../dataset/FB15k-237/entity2textlong.txt")
parser.add_argument("--relation_mapping_path", help="path to the relation2text mapping", default="../dataset/FB15k-237/relaion2text.json")
parser.add_argument("--mongodb_port", help="port of the mongodb collection with the dataset", type=int, default=27017)
parser.add_argument("--input_db", help="name of the mongo database that stores wikidata5m dataset", default='FB15k-237-image')
parser.add_argument("--train_collection_input", help="name of the collection that stores train KG", default='train-set')
parser.add_argument("--valid_collection_input", help="name of the collection that stores valid KG", default='valid-set')
parser.add_argument("--test_collection_input", help="name of the collection that stores test KG", default='test-set')
# parser.add_argument("--figure_collection_input", help="name of the collection that stores test KG", default='figure-set')
parser.add_argument("--train_collection_output", help="name of the collection that stores verbalized train KG", default='verbalized_train')
parser.add_argument("--valid_collection_output", help="name of the collection that stores verbalized valid KG", default='verbalized_valid')
parser.add_argument("--test_collection_output", help="name of the collection that stores verbalized test KG", default='verbalized_test')
# parser.add_argument("--figure_collection_output", help="name of the collection that stores verbalized test KG", default='verbalized_figure')
args = parser.parse_args()
vecs = np.load(args.relation_vectors_path)  #(237, 300)
similarity_matrix = cosine_similarity(vecs)#计算关系向量之间的余弦相似度，得到相似度矩阵 (237, 237)

with open(args.rel2ind_path, 'r',encoding='utf-8') as f:
    rel2ind = json.load(f)  #18
    rel2ind = {key: value for key, value in rel2ind.items() if "inverse of" not in key}

entity_mapping = {}
with open(args.entity_mapping_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        _id, name = line[0], line[1]
        entity_mapping[_id] = name  #entity2text number of entities is 14951

entity_description = {}
with open(args.entity_description_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        _id, name = line[0], line[1]
        if len(name.split('.')) > 3:
            entity_description[_id] = '.'.join(name.split('.')[:3]) + "."
        else:
            entity_description[_id] = '.'.join(name.split('.')[:3])
# max_word_count = max(len(value.split()) for value in entity_description.values())
# print(max_word_count)

with open(args.relation_mapping_path, 'r', encoding='utf-8') as f:
    relation_mapping = json.load(f) #relation2text number of relations is 474
    relation_mapping = {key: value for key, value in relation_mapping.items() if "inverse of" not in key}

training_path = "../dataset/FB15k-237/train.tsv"
validation_path = "../dataset/FB15k-237/dev.tsv"
testing_path = "../dataset/FB15k-237/test.tsv"
# figure_path = "../dataset/WN18/figure.tsv"

train_df = pd.read_csv(training_path, sep='\t', names = ['head', 'relation', 'tail'], encoding='utf-8', dtype={'head': str, 'relation': str, 'tail': str},nrows=20)
valid_df = pd.read_csv(validation_path, sep='\t', names = ['head', 'relation', 'tail'], encoding='utf-8', dtype={'head': str, 'relation': str, 'tail': str},nrows=20)
test_df = pd.read_csv(testing_path, sep='\t', names = ['head', 'relation', 'tail'], encoding='utf-8', dtype={'head': str, 'relation': str, 'tail': str}, nrows=20)
# figure_df = pd.read_csv(figure_path, sep='\t', names = ['head', 'relation', 'tail'], encoding='utf-8', dtype={'head': str, 'relation': str, 'tail': str})
client = MongoClient('localhost', args.mongodb_port)
DB_NAME = args.input_db

#分别获取训练集、验证集、测试集的collection
collection_train = client[DB_NAME][args.train_collection_input]
collection_valid = client[DB_NAME][args.valid_collection_input]
collection_test = client[DB_NAME][args.test_collection_input]
# collection_figure = client[DB_NAME][args.figure_collection_input]

logger.info('Creating indexes in the collection with the KGs...')
#遍历三个collection，分别创建索引
for coll in [collection_train, collection_valid, collection_test]:
    collection_train.create_index([("head", 1)])
    coll.create_index([("tail", 1)])
    coll.create_index([("relation", 1)])


logger.info('Populating collection with the train KG...')
#遍历train_df，将数据转化为dict格式，并插入collection_train中
# docs = []
# for i, doc in tqdm(train_df.iterrows(), total=len(train_df)):
#     docs.append({'_id': i , 'head': doc['head'], 'tail': doc['tail'], 'relation': doc['relation']})
#     if i % 100000 == 0 and i > 0:
#         collection_train.insert_many(docs)
#         docs = []
# collection_train.insert_many(docs)
#
# #分别遍历valid_df和test_df，将数据转化为dict格式，并插入collection_valid和collection_test中
# logger.info('Populating collection with the valid KG...')
# for i, doc in tqdm(valid_df.iterrows(), total=len(valid_df)):
#     collection_valid.insert_one({'_id': i , 'head': doc['head'], 'tail': doc['tail'], 'relation': doc['relation']})
#
#
# logger.info('Populating collection with the test KG...')
# for i, doc in tqdm(test_df.iterrows(), total=len(test_df)):
#     collection_test.insert_one({'_id': i , 'head': doc['head'], 'tail': doc['tail'], 'relation': doc['relation']})
# logger.info('Populating collection with the test KG...')
# for i, doc in tqdm(figure_df.iterrows(), total=len(figure_df)):
#     collection_figure.insert_one({'_id': i , 'head': doc['head'], 'tail': doc['tail'], 'relation': doc['relation']})

#将KG中的实体和关系转化为文本描述
verbalizer_train = Verbalizer(collection_train, similarity_matrix=similarity_matrix,
                                     relation2index=rel2ind, entity2text=entity_mapping, entity_description=entity_description, relation2text=relation_mapping)

def read_dict(dict_path: str):
    """
    Read entity / relation dict.
    Format: dict({id: entity / relation})
    """

    element_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            id_, element= line.strip().split('\t')
            element_dict[element] = int(id_)

    return element_dict

entity_dict_path = '../dataset/FB15k-237/entities.dict'
entity_dict = read_dict(entity_dict_path)


logger.info('Verbalizing train KG...')
ouput_train_collection = client[DB_NAME][args.train_collection_output]
train_json_path = "fb15k-237-verbalizer_update-neighyes/train_verbalized_update_1.json"
# train_res = verbalize_dataset(train_df, ouput_train_collection, verbalizer_train, json_file_path=train_json_path)
# assert train_res == len(train_df) * 2
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
# 运行异步主函数
loop = asyncio.get_event_loop()
train_res = loop.run_until_complete(verbalize_dataset_async(train_df, verbalizer_train, json_file_path=train_json_path))

# logger.info('Verbalizing valid KG...')
# ouput_valid_collection = client[DB_NAME][args.valid_collection_output]
# valid_json_path = "fb15k-237-verbalizer_update/valid_verbalized_update.json"
# valid_res = verbalize_dataset(valid_df, ouput_valid_collection, verbalizer_train, json_file_path=valid_json_path)
# assert valid_res == len(valid_df) * 2

logger.info('Verbalizing test KG...')
ouput_test_collection = client[DB_NAME][args.test_collection_output]
test_json_path = "fb15k-237-verbalizer_update-neighyes/test_verbalized_update_1.json"
# test_res = verbalize_dataset(test_df, ouput_test_collection, verbalizer_train, json_file_path=test_json_path)
test_res = loop.run_until_complete(verbalize_dataset_async(test_df, verbalizer_train, json_file_path=test_json_path))
# assert test_res == len(test_df) * 2
# ouput_test_collection = client[DB_NAME][args.figure_collection_output]
# test_json_path = "wn18-verbalizer_data/figure_verbalized_update.json"
# test_res = verbalize_dataset(figure_df, ouput_test_collection, verbalizer_train, json_file_path=test_json_path)
