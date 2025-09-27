#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
# from kge_adapter import Adapter
# from process_kge import load_pretrain_kge
def load_pretrain_kge(path):
    if "complex" in path:
        return load_complex_model(path)
    kge_model = torch.load(path)
    ent_embs = torch.tensor(kge_model["entity_embedding"]).cpu()
    rel_embs = torch.tensor(kge_model["relation_embedding"]).cpu()
    ent_embs.requires_grad = False
    rel_embs.requires_grad = False
    # ent_dim = ent_embs.shape[1]
    # rel_dim = rel_embs.shape[1]
    # print(ent_dim, rel_dim)
    # if ent_dim != rel_dim:
    #     rel_embs = torch.cat((rel_embs, rel_embs), dim=-1)
    # print(ent_embs.shape, rel_embs.shape)
    # print(ent_embs.requires_grad, rel_embs.requires_grad)
    return ent_embs, rel_embs
def load_complex_model(path):
    kge_model = torch.load(path)
    ent_embs1 = torch.tensor(kge_model["ent_re_embeddings"]).cpu()
    ent_embs2 = torch.tensor(kge_model["ent_im_embeddings"]).cpu()
    rel_embs1 = torch.tensor(kge_model["rel_re_embeddings"]).cpu()
    rel_embs2 = torch.tensor(kge_model["rel_im_embeddings"]).cpu()
    ent_embs = torch.cat((ent_embs1, ent_embs2), dim=-1)
    rel_embs = torch.cat((rel_embs1, rel_embs2), dim=-1)
    ent_embs.requires_grad = False
    rel_embs.requires_grad = False
    return ent_embs, rel_embs

class PretrainKGEmbedding(nn.Module):  # 结构嵌入预训练，捕获KG中的结构信息，返回prefix
    def __init__(
            self,
            pretrain_ent_embs,
            pretrain_rel_embs,
            dim_llm,
            num_prefix
    ):
        super(PretrainKGEmbedding, self).__init__()
        self.num_prefix = num_prefix    # num_prefix = 1
        self.llm_dim = dim_llm  # 4096
        self.emb_dim = num_prefix * dim_llm
        self.ent_embeddings = nn.Embedding.from_pretrained(pretrain_ent_embs)
        self.rel_embeddings = nn.Embedding.from_pretrained(pretrain_rel_embs)
        self.pretrain_dim = self.ent_embeddings.weight.shape[1]
        # Froze the pretrain embeddings
        self.ent_embeddings.requires_grad_(False)
        self.rel_embeddings.requires_grad_(False)
        self.adapter = nn.Linear(self.pretrain_dim, self.emb_dim)

    def forward(self, triple_ids):
        # main training stage
        if triple_ids.shape[1] == 3:
            head, relation, tail = triple_ids[:, 0], triple_ids[:, 1], triple_ids[:, 2]
            h = self.ent_embeddings(head)
            r = self.rel_embeddings(relation)
            t = self.ent_embeddings(tail)
            # pretrain_embs = torch.stack((h, r, t), dim=1)  # 张量堆叠
            prefix_h = nn.Linear(h.shape[1], self.llm_dim)
            prefix_r = nn.Linear(r.shape[1], self.llm_dim)
            prefix_t = nn.Linear(t.shape[1], self.llm_dim)
            prefix_pretrain_embs = torch.cat((prefix_h, prefix_r, prefix_t), dim=1)
            prefix = prefix_pretrain_embs.reshape(-1, 3*self.num_prefix, self.llm_dim)
            return prefix
        # entity-aware pre-funing
        else:
            ent = triple_ids.reshape(-1, )
            emb = self.ent_embeddings(ent)
            prefix = self.adapter(emb).reshape(-1, self.num_prefix, self.llm_dim)
            return prefix
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, num_prefix: int=1, pretrain_emb_path=None):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)#模型实例化
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 优化加载：如果在预训练阶段已经加载过，可以将嵌入缓存到内存中
        if hasattr(self, 'cached_ent_embs') and hasattr(self, 'cached_rel_embs'):
            print("Using cached embeddings.")
            ent_embs, rel_embs = self.cached_ent_embs, self.cached_rel_embs
        else:
            print("Loading pre-trained embeddings from disk.")
            # Load embeddings and cache them in the instance
            # ent_embs, rel_embs = load_pretrain_kge("/root/autodl-tmp/IT-MKGC/kge_models/wn18-embeddings.pth")
            ent_embs, rel_embs = load_pretrain_kge("E:\MMKG\KGC\IT-MKGC\kge_models\wn18-embeddings.pth")
            self.cached_ent_embs = ent_embs
            self.cached_rel_embs = rel_embs

        if pretrain_emb_path is None:  # 预训练适配器
            print("Adapter Trained From Scratch".format(pretrain_emb_path))
            self.embeddings = PretrainKGEmbedding(
                pretrain_ent_embs=ent_embs,
                pretrain_rel_embs=rel_embs,
                dim_llm=4096,
                num_prefix=num_prefix
            )
        else:
            print("Adapter Load From {}".format(pretrain_emb_path))
            self.embeddings = torch.load(pretrain_emb_path)


        # Initialize weights and apply final processing
        self.post_init()

        # self.adapter = Adapter(self.model, num_prefix=config.num_prefix, kge_model=config.kge_model)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        embedding_ids: Optional[torch.LongTensor] = None,#结构信息的embedding ID
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        kg_embeds = self.embeddings(embedding_ids)  #[batch_size, num_prefix, dim_llm] = [16, 1, 4096]

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                kg_embeds,
                labels,
                images,
                image_sizes
            )


        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        embedding_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None) #none
        attention_mask = kwargs.pop("attention_mask", None)#none

        kg_embeds = self.embeddings(embedding_ids)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # if images is not None:
        #     (
        #         inputs,
        #         position_ids,
        #         attention_mask,
        #         _,
        #         inputs_embeds,#torch.float16
        #         _
        #     ) = self.prepare_inputs_labels_for_multimodal(
        #         inputs,
        #         position_ids,
        #         attention_mask,
        #         None,
        #         kg_embeds,
        #         None,
        #         images,#torch.float16
        #         image_sizes=image_sizes#(224, 168)
        #     )
        # else:
        #     inputs_embeds = self.get_model().embed_tokens(inputs)
        #     inputs_embeds = torch.cat((kg_embeds, inputs_embeds), dim=1)

        inputs, position_ids, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
            inputs,
            position_ids,
            attention_mask,
            None,
            kg_embeds,
            None,
            images,#torch.float16
            image_sizes=image_sizes#(224, 168)
        )

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            embedding_ids = embedding_ids,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        embedding_ids = kwargs.get("embedding_ids", None)

        # kg_embeds = kwargs.pop("kg_embeds", None)


        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        inputs.pop("cache_position")
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if embedding_ids is not None:  # 将 embedding_ids 加入 inputs
            inputs["embedding_ids"] = embedding_ids
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
