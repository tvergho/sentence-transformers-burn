import pathlib
import torch
import numpy as np
from transformers import BertModel

def save_scalar(s, name, path):
    s = np.array([1.0, float(s)]).astype(np.float32)
    np.save(pathlib.Path(path, f'{name}.npy'), s)

def save_tensor(tensor, name, path):
    tensor_numpy = tensor.numpy()
    tensor_dims = np.array(tensor_numpy.shape) 
    tensor_values = tensor_numpy.flatten()
    tensor_to_save = np.concatenate((tensor_dims, tensor_values)).astype(np.float32)
    np.save(pathlib.Path(path, f'{name}.npy'), tensor_to_save)

def save_linear(linear, name, path):
    path = pathlib.Path(path, name)
    path.mkdir(parents=True, exist_ok=True)
    save_tensor(linear.weight.t(), 'weight', path)
    if linear.bias is not None:
        save_tensor(linear.bias, 'bias', path)
    # Save in and out features
    save_scalar(linear.in_features, 'in_features', path)
    save_scalar(linear.out_features, 'out_features', path)

def save_bert_self_attention(self_attention, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(self_attention.query, 'query', path)
    save_linear(self_attention.key, 'key', path)
    save_linear(self_attention.value, 'value', path)

def save_bert_self_output(self_output, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(self_output.dense, 'dense', path)
    save_tensor(self_output.LayerNorm.weight, 'LayerNorm_weight', path)
    save_tensor(self_output.LayerNorm.bias, 'LayerNorm_bias', path)

def save_bert_attention(attention, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_bert_self_attention(attention.self, pathlib.Path(path, 'self'))
    save_bert_self_output(attention.output, pathlib.Path(path, 'output'))

def save_bert_intermediate(intermediate, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(intermediate.dense, 'dense', path)

def save_bert_output(output, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(output.dense, 'dense', path)
    save_tensor(output.LayerNorm.weight, 'LayerNorm_weight', path)
    save_tensor(output.LayerNorm.bias, 'LayerNorm_bias', path)

def save_bert_layer(layer, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_bert_attention(layer.attention, pathlib.Path(path, 'attention'))
    save_bert_intermediate(layer.intermediate, pathlib.Path(path, 'intermediate'))
    save_bert_output(layer.output, pathlib.Path(path, 'output'))

def save_bert_encoder(encoder, path):
    with torch.no_grad():
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        save_scalar(len(encoder.layer), 'n_layer', path)
        for idx, layer in enumerate(encoder.layer):
            save_bert_layer(layer, pathlib.Path(path, f'layer{idx}'))

def save_embedding(embedding, name, path):
    path = pathlib.Path(path, name)
    path.mkdir(parents=True, exist_ok=True)
    save_tensor(embedding.weight, 'weight', path)
    save_scalar(embedding.num_embeddings, 'num_embeddings', path)
    save_scalar(embedding.embedding_dim, 'embedding_dim', path)

def save_bert_embeddings(embeddings, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_embedding(embeddings.word_embeddings, 'word_embeddings', path)
    save_embedding(embeddings.position_embeddings, 'position_embeddings', path)
    save_embedding(embeddings.token_type_embeddings, 'token_type_embeddings', path)
    save_tensor(embeddings.LayerNorm.weight, 'LayerNorm_weight', path)
    save_tensor(embeddings.LayerNorm.bias, 'LayerNorm_bias', path)

if __name__ == "__main__":
  model = BertModel.from_pretrained('BAAI/bge-small-en')
  model.requires_grad_(False)
  save_bert_encoder(model.encoder, 'model/encoder')
  save_bert_embeddings(model.embeddings, 'model/embeddings')
  print("Saved encoder to encoder/ and embeddings to model/")