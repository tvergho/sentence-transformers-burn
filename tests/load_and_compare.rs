extern crate sentence_transformers;

use sentence_transformers::bert_loader::load_model;
use sentence_transformers::model::bert_embeddings::BertEmbeddingsInferenceBatch;
use sentence_transformers::model::bert_model::{BertModel, BertModelConfig};
use burn_tch::{TchDevice, TchBackend};
use burn::tensor::{Tensor, Shape};
use npy::NpyData;
use std::io::Read;

fn setup() -> String {
    // Get the current directory
    let mut path = std::env::current_dir().unwrap();

    // Append the relative path to the tests directory
    path.push("tests");
    path.push("model");

    // Check if model has been generated in model/ directory
    // If not, panic and return
    if !path.exists() {
        panic!("Model not found in directory: {:?}. Run python scripts/prepare_test.py to generate a test model first.", path);
    }

    // Remove model directory from path
    path.pop();

    path.to_str().unwrap().to_string()
}

#[test]
fn compare_dump_model_outputs() {
  let path = setup();
  let device = TchDevice::Cpu;
  let config = BertModelConfig { 
    n_heads: 6, 
    n_layers: 6, 
    layer_norm_eps: 1e-12, 
    hidden_size: 192, 
    intermediate_size: 768, 
    hidden_act: "gelu".to_string(),
    vocab_size: 30522, 
    max_position_embeddings: 512, 
    type_vocab_size: 2, 
    hidden_dropout_prob: 0.1 
  };

  let model: BertModel<TchBackend<f32>> = load_model((path.clone() + "/model").as_str(), &device, config);
  let input_ids: Tensor<TchBackend<f32>, 1> = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).to_device(&device.clone());
  let attn_mask: Tensor<TchBackend<f32>, 1> = Tensor::from_floats([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to_device(&device.clone());

  let input_ids = input_ids.int().reshape(Shape::new([1, 10])); 
  let attn_mask = attn_mask.reshape(Shape::new([1, 10]));

  println!("input_ids: {:?}", input_ids.clone().to_data().value);
  println!("attn_mask: {:?}", attn_mask.clone().to_data().value);
  
  let input = BertEmbeddingsInferenceBatch {
    tokens: input_ids,
    mask_attn: Some(attn_mask),
  };

  let output = model.forward(input);
  let output_vec: Vec<f32> = output.to_data().value.to_vec();

  // Compare with outputs from outputs/ directory
  let output_path = path + "/outputs/outputs_0.npy";
  let mut buf = vec![];
    std::fs::File::open(output_path).unwrap()
        .read_to_end(&mut buf).unwrap();

  let data: NpyData<f32> = NpyData::from_bytes(&buf).unwrap();
  let data_vec: Vec<f32> = data.to_vec();

  // Compare length of each
  assert_eq!(output_vec.len(), data_vec.len());

  // Compare each value
  for i in 0..output_vec.len() {
    assert!((output_vec[i] - data_vec[i]).abs() < 0.0001);
  }
}