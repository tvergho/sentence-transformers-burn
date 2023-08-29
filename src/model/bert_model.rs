use burn::{
  config::Config,
  module::Module,
  tensor::{backend::Backend, Tensor},
};
use super::bert_embeddings::{BertEmbeddings, BertEmbeddingsConfig, BertEmbeddingsInferenceBatch};
use super::bert_encoder::{BertEncoder, BertEncoderConfig, BertEncoderInput};

// Define the Bert model configuration
#[derive(Config)]
pub struct BertModelConfig {
  /// The number of attention heads.
  pub n_heads: usize,
  /// The number of layers.
  pub n_layers: usize,
  pub layer_norm_eps: f64,
  pub hidden_size: usize,
  pub intermediate_size: usize,
  pub hidden_act: String,
  pub vocab_size: usize,
  pub max_position_embeddings: usize,
  pub type_vocab_size: usize,
  pub hidden_dropout_prob: f64,
}

// Define the Bert model structure
#[derive(Module, Debug)]
pub struct BertModel<B: Backend> {
  pub encoder: BertEncoder<B>,
  pub embeddings: BertEmbeddings<B>,
}

impl BertModelConfig {
    /// Initializes a Bert model with default weights
    pub fn init<B: Backend>(&self) -> BertModel<B> {
      let embeddings = BertEmbeddingsConfig {
        vocab_size: self.vocab_size,
        max_position_embeddings: self.max_position_embeddings,
        type_vocab_size: self.type_vocab_size,
        hidden_size: self.hidden_size,
        hidden_dropout_prob: self.hidden_dropout_prob,
        layer_norm_eps: self.layer_norm_eps,
      }.init();
      let encoder = BertEncoderConfig {
        n_heads: self.n_heads,
        n_layers: self.n_layers,
        dropout: self.hidden_dropout_prob,
        layer_norm_eps: self.layer_norm_eps,
        hidden_size: self.hidden_size,
        intermediate_size: self.intermediate_size,
        hidden_act: "gelu".to_string(),
      }.init();

      BertModel {
        encoder,
        embeddings,
      }
    }

    /// Initializes a Bert model with provided weights
    pub fn init_with<B: Backend>(&self, record: BertModelRecord<B>) -> BertModel<B> {
      let embeddings = BertEmbeddingsConfig {
        vocab_size: self.vocab_size,
        max_position_embeddings: self.max_position_embeddings,
        type_vocab_size: self.type_vocab_size,
        hidden_size: self.hidden_size,
        hidden_dropout_prob: self.hidden_dropout_prob,
        layer_norm_eps: self.layer_norm_eps,
      }.init_with(record.embeddings);
      let encoder = BertEncoderConfig {
        n_heads: self.n_heads,
        n_layers: self.n_layers,
        dropout: self.hidden_dropout_prob,
        layer_norm_eps: self.layer_norm_eps,
        hidden_size: self.hidden_size,
        intermediate_size: self.intermediate_size,
        hidden_act: "gelu".to_string(),
      }.init_with(record.encoder);

      BertModel {
          encoder,
          embeddings,
      }
    }
}

impl<B: Backend> BertModel<B> {
    /// Defines forward pass
    pub fn forward(&self, input: BertEmbeddingsInferenceBatch<B>) -> Tensor<B, 3> {
        let embedding = self.embeddings.forward(input.clone());

        let shape = input.tokens.shape();
        let mut mask_attn: Tensor<B, 2> = Tensor::ones(shape.clone()).to_device(&input.tokens.device());
        if input.mask_attn.is_some() {
          mask_attn = input.mask_attn.unwrap();
        }

        let extended_mask_attn = self.get_extended_attention_mask(mask_attn, shape.dims);

        let encoder_input = BertEncoderInput::new(embedding, extended_mask_attn);
        let output = self.encoder.forward(encoder_input);
        output
    }

  pub fn get_extended_attention_mask(
      &self,
      attention_mask: Tensor<B, 2>,
      input_shape: [usize; 2],
  ) -> Tensor<B, 4> {
    // Handling attention_mask.dim() == 3 case:
    // If attention_mask is 3D, just expand the dimension
    if attention_mask.dims().len() == 3 {
      return attention_mask.unsqueeze::<4>();
    }

    let device = attention_mask.device();

    let mut extended_attention_mask: Tensor<B, 4>;
    extended_attention_mask = attention_mask.clone().unsqueeze::<3>().unsqueeze::<4>().to_device(&device.clone());
    extended_attention_mask = extended_attention_mask.reshape([input_shape[0], 1, 1, input_shape[1]]).to_device(&device.clone());
    
    let min_val = f32::MIN;
    extended_attention_mask = (Tensor::<B, 4>::ones(extended_attention_mask.shape()).to_device(&device.clone()).sub(extended_attention_mask)).mul_scalar(min_val);
    extended_attention_mask
  }
}