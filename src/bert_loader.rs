use burn::{
  module::{Module, Param, ParamId, ConstantRecord},
  nn::LayerNormRecord,
  nn::{LinearRecord, EmbeddingRecord},
  tensor::{backend::Backend, Tensor, Data, Shape},
};
use crate::model::{
  bert_encoder::{BertEncoderRecord, BertEncoderLayerRecord, BertOutputRecord, BertIntermediateRecord, BertAttentionRecord, BertSelfAttentionRecord, BertSelfOutputRecord}, 
  bert_embeddings::BertEmbeddingsRecord,
  bert_model::{BertModelConfig, BertModelRecord, BertModel}
};
use std::io::Read;
use npy::NpyData;

fn load_npy_scalar<B: Backend>(filename: &str) -> Vec<f32> {
    // Open the file in read-only mode.
    let mut buf = vec![];

    std::fs::File::open(filename).unwrap()
        .read_to_end(&mut buf).unwrap();

    let data: NpyData<f32> = NpyData::from_bytes(&buf).unwrap();
    let data_vec: Vec<f32> = data.to_vec();
    data_vec
}

fn load_1d_tensor<B: Backend>(filename: &str) -> Param<Tensor<B, 1>> {
  // Open the file in read-only mode.
  let mut buf = vec![];

  std::fs::File::open(filename).unwrap()
      .read_to_end(&mut buf).unwrap();

  let data: NpyData<f32> = NpyData::from_bytes(&buf).unwrap();
  let data_vec: Vec<f32> = data.to_vec();

  let tensor_length = data_vec[0];
  let shape = Shape::new([tensor_length as usize]);
  let data = Data::new(data_vec[1..].to_vec(), shape);

  // Convert the loaded data into an actual 1D Tensor using from_floats
  let tensor = Tensor::<B, 1>::from_floats(data);

  let param = Param::new(ParamId::new(), tensor);
  param
}

fn load_2d_tensor<B: Backend>(filename: &str) -> Param<Tensor<B, 2>> {
  // Open the file in read-only mode.
  let mut buf = vec![];

  std::fs::File::open(filename).unwrap()
      .read_to_end(&mut buf).unwrap();

  let data: NpyData<f32> = NpyData::from_bytes(&buf).unwrap();
  let data_vec: Vec<f32> = data.to_vec();

  // Extract tensor shape from the beginning of the loaded array.
  // Here we're assuming that we've saved the tensor's shape as 2 f32 values at the start.
  let tensor_rows: usize = data_vec[0] as usize;
  let tensor_cols: usize = data_vec[1] as usize;

  // Extract tensor values
  let tensor_values_flat: Vec<f32> = data_vec[2..].to_vec(); 

  // Convert the reshaped data into an actual 2D Tensor using from_floats
  let shape = Shape::new([tensor_rows, tensor_cols]);
  let data = Data::new(tensor_values_flat, shape);
  let tensor = Tensor::<B, 2>::from_floats(data);

  let param = Param::new(ParamId::new(), tensor);
  param
}

fn load_layer_norm<B: Backend>(dir: &str) -> LayerNormRecord<B> {
  let layer_norm_record = LayerNormRecord {
    beta: load_1d_tensor::<B>(&format!("{}LayerNorm_bias.npy", dir)),
    gamma: load_1d_tensor::<B>(&format!("{}LayerNorm_weight.npy", dir)),
    epsilon: ConstantRecord::new()
  };
  layer_norm_record
}

fn load_linear<B: Backend>(dir: &str) -> LinearRecord<B> {
  let linear_record = LinearRecord {
    weight: load_2d_tensor::<B>(&format!("{}weight.npy", dir)),
    bias: Some(load_1d_tensor::<B>(&format!("{}bias.npy", dir))),
  };
  linear_record
}

fn load_output_layer<B: Backend>(layer_dir: &str) -> BertOutputRecord<B> {
  let output_record = BertOutputRecord {
    dense: load_linear::<B>(&format!("{}dense/", layer_dir)),
    layer_norm: load_layer_norm::<B>(&format!("{}", layer_dir)),
    dropout: ConstantRecord::new()
  };
  output_record
}

fn load_intermediate_layer<B: Backend>(layer_dir: &str) -> BertIntermediateRecord<B> {
  let intermediate_record = BertIntermediateRecord {
    dense: load_linear::<B>(&format!("{}dense/", layer_dir)),
    intermediate_act: ConstantRecord::new(),
  };
  intermediate_record
}

fn load_self_attention_layer<B: Backend>(layer_dir: &str) -> BertSelfAttentionRecord<B> {
  let attention_record = BertSelfAttentionRecord {
    query: load_linear::<B>(&format!("{}query/", layer_dir)),
    key: load_linear::<B>(&format!("{}key/", layer_dir)),
    value: load_linear::<B>(&format!("{}value/", layer_dir)),
    dropout: ConstantRecord::new(),
    num_attention_heads: ConstantRecord::new(),
    attention_head_size: ConstantRecord::new(),
    all_head_size: ConstantRecord::new(),
  };
  attention_record
}

fn load_self_output_layer<B: Backend>(layer_dir: &str) -> BertSelfOutputRecord<B> {
  let output_record = BertSelfOutputRecord {
    dense: load_linear::<B>(&format!("{}dense/", layer_dir)),
    layer_norm: load_layer_norm::<B>(&format!("{}", layer_dir)),
    dropout: ConstantRecord::new()
  };
  output_record
}

fn load_attention_layer<B: Backend>(layer_dir: &str) -> BertAttentionRecord<B> {
  let attention_record = BertAttentionRecord {
    self_attention: load_self_attention_layer::<B>(&format!("{}self/", layer_dir)),
    self_output: load_self_output_layer::<B>(&format!("{}output/", layer_dir)),
  };
  attention_record
}

fn load_encoder<B: Backend>(encoder_dir: &str) -> BertEncoderRecord<B> {
  // Load n_layer
  let n_layer = load_npy_scalar::<B>(&format!("{}n_layer.npy", encoder_dir));
  let num_layers = n_layer[1] as usize;

  // Load layers
  let mut layers: Vec<BertEncoderLayerRecord<B>> = Vec::new();
  
  for i in 0..num_layers {
    let layer_dir = format!("{}layer{}/", encoder_dir, i);
    let attention_layer = load_attention_layer::<B>(format!("{}attention/", layer_dir).as_str());
    let intermediate_layer = load_intermediate_layer::<B>(format!("{}intermediate/", layer_dir).as_str());
    let output_layer = load_output_layer::<B>(format!("{}output/", layer_dir).as_str());

    let layer_record = BertEncoderLayerRecord {
      attention: attention_layer,
      intermediate: intermediate_layer,
      output: output_layer,
    };

    layers.push(layer_record);
  }

  let encoder_record = BertEncoderRecord {
    layers,
  };

  encoder_record
}

fn load_embedding<B: Backend>(embedding_dir: &str) -> EmbeddingRecord<B> {
  let embedding = EmbeddingRecord {
    weight: load_2d_tensor::<B>(&format!("{}weight.npy", embedding_dir)),
  };

  embedding
}

fn load_embeddings<B: Backend>(embeddings_dir: &str) -> BertEmbeddingsRecord<B> {
  let word_embeddings = load_embedding::<B>(&format!("{}word_embeddings/", embeddings_dir));
  let position_embeddings = load_embedding::<B>(&format!("{}position_embeddings/", embeddings_dir));
  let token_type_embeddings = load_embedding::<B>(&format!("{}token_type_embeddings/", embeddings_dir));
  let layer_norm = load_layer_norm::<B>(&format!("{}", embeddings_dir));
  let dropout = ConstantRecord::new();

  let embeddings_record = BertEmbeddingsRecord {
    word_embeddings,
    position_embeddings,
    token_type_embeddings,
    layer_norm,
    dropout,
    max_position_embeddings: ConstantRecord::new(),
  };

  embeddings_record
}

pub fn load_model<B: Backend>(dir: &str, device: &B::Device, config: BertModelConfig) -> BertModel<B> {
  let encoder_record = load_encoder::<B>(&format!("{}/encoder/", dir));
  let embeddings_record = load_embeddings::<B>(&format!("{}/embeddings/", dir));

  let model_record = BertModelRecord {
      embeddings: embeddings_record,
      encoder: encoder_record,
  };

  let mut model = config.init_with::<B>(model_record);

  model = model.to_device(device);
  model
}