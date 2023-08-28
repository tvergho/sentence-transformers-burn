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
  pub encoder: BertEncoderConfig,
  pub config: BertEmbeddingsConfig,
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
        let encoder = self.encoder.init();
        let embeddings = self.config.init();

        BertModel {
          encoder,
          embeddings,
        }
    }

    /// Initializes a Bert model with provided weights
    pub fn init_with<B: Backend>(&self, record: BertModelRecord<B>) -> BertModel<B> {
        let encoder = self.encoder.init_with(record.encoder);
        let embeddings = self.config.init_with(record.embeddings);

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