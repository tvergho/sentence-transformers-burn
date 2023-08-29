import torch
from transformers import BertModel, BertConfig
from safetensors.torch import save_file
import os
from dump import save_bert_encoder, save_bert_embeddings
import numpy as np

config = BertConfig(
  vocab_size=30522,
  hidden_size=384,
  num_hidden_layers=12,
  num_attention_heads=12,
  intermediate_size=1536,
  hidden_act="gelu",
  hidden_dropout_prob=0.1,
  layer_norm_eps=1e-12,
  max_position_embeddings=512,
)

model = BertModel(config)
model = BertModel.from_pretrained("BAAI/bge-small-en")
model.requires_grad_(False)
model.eval()

# Save the models in two formats
if not os.path.exists("tests/model"):
  os.mkdir("tests/model")

# First, save the model in the standard safetensors format
save_file(model.state_dict(), "tests/model/bert_model.safetensors")
config.to_json_file("tests/model/bert_config.json")

# Second, save the model in the format used by the dump.py script
save_bert_encoder(model.encoder, 'tests/model/encoder')
save_bert_embeddings(model.embeddings, 'tests/model/embeddings')

# Next, save some example outputs
if not os.path.exists("tests/outputs"):
  os.mkdir("tests/outputs")

# Create some dummy inputs
input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

# Save the outputs of the model as numpy array
outputs = model(input_ids, attention_mask=attention_mask)
np.save("tests/outputs/outputs_0.npy", outputs[0].detach().flatten().numpy())