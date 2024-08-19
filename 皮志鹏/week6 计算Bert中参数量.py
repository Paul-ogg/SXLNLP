"""
计算Bert参数量（一层Transformer）:

embedding层（token_embedding + segment_embedding + position_embedding + add&normalize）+
Transformer层（self-attention + feedforward ） + 
pool_fc

self-attention = attention + liner + add&normalize
feedforward = liner1 + liner2 + add&normalize
"""

vocab_size = 21128
embedding_size = 768
segment_size = 2
position_size = 512
intermediate_size = embedding_size * 4  #前馈层维度 

token_embedding_params = vocab_size * embedding_size
segment_embedding_params = segment_size * embedding_size
position_embedding_params = position_size * embedding_size
add_normalize = embedding_size + embedding_size
embedding_parameters = token_embedding_params + segment_embedding_params + position_embedding_params + add_normalize


attention = (embedding_size * embedding_size + embedding_size) * 3  # 计算Q K V各一组参数
liner = embedding_size * embedding_size + embedding_size
self_attention = attention + liner + add_normalize

liner1 = embedding_size * intermediate_size + intermediate_size
liner2 = intermediate_size * embedding_size + embedding_size
feedforward = liner1 + liner2 + add_normalize

transformer_parameters = self_attention + feedforward 

pool_fc = embedding_size * embedding_size + embedding_size

model_parameters = embedding_parameters + transformer_parameters + pool_fc

from transformers import BertModel

model = BertModel.from_pretrained(r"E:\BaiduNetdiskDownload\week6 预训练模型\bert-base-chinese", return_dict=False)

print(f"Bert中参数量为{sum(p.numel() for p in model.parameters())} \n 计算参数量为{model_parameters}")
