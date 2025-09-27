import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
import torch.fx
from transformers import LlamaForCausalLM
# from ..model import LlavaLlamaForCausalLM
# from llava.model import *
from llava_llama import LlavaLlamaForCausalLM
from process_kge import load_pretrain_kge

class Adapter(nn.Module):  # 适配器
    def __init__(
            self,
            model: LlamaForCausalLM,
            num_prefix: int,
            kge_model: str = "data/UMLS-rotate.pth",
            pretrain_emb_path=None
    ) -> None:
        super(Adapter, self).__init__()
        self.llama_model = model
        ent_embs, rel_embs = load_pretrain_kge(kge_model)  # 加载预训练的结构嵌入
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
            return_dict: Optional[bool] = None,
            embedding_ids: torch.LongTensor = None
    ):
        kg_embeds = self.embeddings(embedding_ids)  # torch.Size([12, 3, 4096])
        # print(kg_embeds.shape)
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.llama_model.model.model.embed_tokens(
            input_ids)  # torch.Size([12, 128, 4096]) input_ids:torch.Size([12, 128])
        #拼接结构嵌入和文本嵌入
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)  # torch.Size([12, 131, 4096]) 将结构嵌入和文本嵌入进行拼接
        #更新注意力掩码和标签
        prefix_mask = torch.ones((batch_size, seq_len))  # torch.Size([12, 3])
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)  # torch.Size([12, 3])

        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)  # torch.Size([12, 131])
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)  # torch.Size([12, 131])
        #将拼接的嵌入和标签作为新的输入
        return self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,  # 模型输入增加前缀
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class PretrainKGEmbedding(nn.Module):  # 结构嵌入预训练，捕获KG中的结构信息，返回prefix
    def __init__(
            self,
            pretrain_ent_embs,
            pretrain_rel_embs,
            dim_llm,
            num_prefix
    ):
        super(PretrainKGEmbedding, self).__init__()
        self.num_prefix = num_prefix
        self.llm_dim = dim_llm
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
            pretrain_embs = torch.stack((h, r, t), dim=1)  # 张量堆叠
            prefix = self.adapter(pretrain_embs).reshape(-1, 3 * self.num_prefix, self.llm_dim)
            return prefix
        # entity-aware pre-funing
        else:
            ent = triple_ids.reshape(-1, )
            emb = self.ent_embeddings(ent)
            prefix = self.adapter(emb).reshape(-1, self.num_prefix, self.llm_dim)
            # print(prefix.shape)
            return prefix

