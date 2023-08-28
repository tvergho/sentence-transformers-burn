use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, GELU
    },
    tensor::{backend::Backend, Tensor, activation},
};
use libm::sqrtf;

#[derive(Config)]
pub struct BertEncoderConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The number of attention heads.
    pub n_heads: usize,
    /// The number of layers.
    pub n_layers: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
    pub layer_norm_eps: f64,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
}

#[derive(Module, Debug)]
pub struct BertEncoder<B: Backend> {
    pub layers: Vec<BertEncoderLayer<B>>,
}

#[derive(Debug)]
pub struct BertEncoderInput<B: Backend> {
    tensor: Tensor<B, 3>,
    mask_attn: Tensor<B, 4>,
}

impl<B: Backend> BertEncoderInput<B> {
    pub fn new(tensor: Tensor<B, 3>, mask_attn: Tensor<B, 4>) -> Self {
        Self {
            tensor,
            mask_attn,
        }
    }
}

impl BertEncoderConfig {
  pub fn init<B: Backend>(&self) -> BertEncoder<B> {
      let layers = (0..self.n_layers)
          .map(|_| BertEncoderLayer::new(self))
          .collect::<Vec<_>>();

      BertEncoder { layers }
  }
  
  pub fn init_with<B: Backend>(
      &self,
      record: BertEncoderRecord<B>,
  ) -> BertEncoder<B> {
      BertEncoder {
          layers: record
              .layers
              .into_iter()
              .map(|record| BertEncoderLayer::new_with(self, record))
              .collect(),
      }
  }
}

impl<B: Backend> BertEncoder<B> {
  pub fn forward(&self, input: BertEncoderInput<B>) -> Tensor<B, 3> {
      let mut x = input.tensor;
      let mask_attn = input.mask_attn;

      for layer in self.layers.iter() {
          x = layer.forward(x, mask_attn.clone());
      }

      x
  }
}

// Define the structure of BertEncoderLayer
#[derive(Debug, Module)]
pub struct BertEncoderLayer<B: Backend> {
  pub attention: BertAttention<B>,
  intermediate: BertIntermediate<B>,
  output: BertOutput<B>,
}

// Constructor and other methods for BertEncoderLayer
impl<B: Backend> BertEncoderLayer<B> {
  pub fn new(config: &BertEncoderConfig) -> Self {
    let attention = BertAttentionConfig {
      hidden_size: config.hidden_size,
      num_attention_heads: config.n_heads,
      attention_head_size: config.hidden_size / config.n_heads,
      layer_norm_eps: config.layer_norm_eps,
      hidden_dropout_prob: config.dropout,
    }.init();
    let intermediate = BertIntermediateConfig {
      hidden_size: config.hidden_size,
      intermediate_size: config.intermediate_size,
    }.init();
    let output = BertOutputConfig {
      intermediate_size: config.intermediate_size,
      hidden_size: config.hidden_size,
      layer_norm_eps: config.layer_norm_eps,
      hidden_dropout_prob: config.dropout,
    }.init();

    BertEncoderLayer {
      attention,
      intermediate,
      output,
    }
  }

  pub fn new_with(config: &BertEncoderConfig, record: BertEncoderLayerRecord<B>) -> Self {
    let attention = BertAttentionConfig {
      hidden_size: config.hidden_size,
      num_attention_heads: config.n_heads,
      attention_head_size: config.hidden_size / config.n_heads,
      layer_norm_eps: config.layer_norm_eps,
      hidden_dropout_prob: config.dropout,
    }.init_with(record.attention);
    let intermediate = BertIntermediateConfig {
      hidden_size: config.hidden_size,
      intermediate_size: config.intermediate_size,
    }.init_with(record.intermediate);
    let output = BertOutputConfig {
      intermediate_size: config.intermediate_size,
      hidden_size: config.hidden_size,
      layer_norm_eps: config.layer_norm_eps,
      hidden_dropout_prob: config.dropout,
    }.init_with(record.output);

    BertEncoderLayer {
      attention,
      intermediate,
      output,
    }
  }

  pub fn forward(&self, x: Tensor<B, 3>, mask_attn: Tensor<B, 4>) -> Tensor<B, 3> {
      let attention_output = self.attention.forward(x, mask_attn);
      let intermediate_output = self.intermediate.forward(attention_output.clone());
      self.output.forward(intermediate_output, attention_output)
  }
}



// BertAttention Configuration and Module
#[derive(Config, Debug)]
pub struct BertAttentionConfig {
  pub hidden_size: usize,         // Dimension of the input
  pub num_attention_heads: usize,     // Number of attention heads
  pub attention_head_size: usize, // Size of each attention head
  pub layer_norm_eps: f64,
  pub hidden_dropout_prob: f64,
}

#[derive(Module, Debug)]
pub struct BertAttention<B: Backend> {
  pub self_attention: BertSelfAttention<B>,
  self_output: BertSelfOutput<B>,
}

impl BertAttentionConfig {
  pub fn init<B: Backend>(&self) -> BertAttention<B> {
    let self_attention = BertSelfAttentionConfig { 
      hidden_size: self.hidden_size,
      num_attention_heads: self.num_attention_heads,
      attention_head_size: self.attention_head_size,
      hidden_dropout_prob: self.hidden_dropout_prob,
    }.init();
    let self_output = BertSelfOutputConfig {
      hidden_size: self.hidden_size,
      layer_norm_eps: self.layer_norm_eps,
      hidden_dropout_prob: self.hidden_dropout_prob,
     }.init();

    BertAttention {
      self_attention,
      self_output,
    }
  }

  pub fn init_with<B: Backend>(&self, record: BertAttentionRecord<B>) -> BertAttention<B> {
    let self_attention = BertSelfAttentionConfig { 
      hidden_size: self.hidden_size,
      num_attention_heads: self.num_attention_heads,
      attention_head_size: self.attention_head_size,
      hidden_dropout_prob: self.hidden_dropout_prob,
    }.init_with(record.self_attention);
    let self_output = BertSelfOutputConfig {
      hidden_size: self.hidden_size,
      layer_norm_eps: self.layer_norm_eps,
      hidden_dropout_prob: self.hidden_dropout_prob,
    }.init_with(record.self_output);

    BertAttention {
      self_attention,
      self_output,
    }
  }
}

impl<B: Backend> BertAttention<B> {
  pub fn forward(&self, hidden_states: Tensor<B, 3>, mask_attn: Tensor<B, 4>) -> Tensor<B, 3> {
    let self_outputs = self.self_attention.forward(hidden_states.clone(), mask_attn);
    let attention_output = self.self_output.forward(self_outputs, hidden_states);
    attention_output
  }
}

// BertIntermediate Configuration and Module
#[derive(Config, Debug)]
pub struct BertIntermediateConfig {
  pub hidden_size: usize,
  pub intermediate_size: usize,
}

#[derive(Module, Debug)]
pub struct BertIntermediate<B: Backend> {
  dense: Linear<B>,
  intermediate_act: GELU,
}

impl BertIntermediateConfig {
    pub fn init<B: Backend>(&self) -> BertIntermediate<B> {
      let dense = LinearConfig::new(self.hidden_size, self.intermediate_size).init();
      let intermediate_act = GELU::new(); // TODO: Change this to HiddenActLayer::new(self.hidden_act) to allow RELU

      BertIntermediate { 
        dense,
        intermediate_act,
      }
    }

    pub fn init_with<B: Backend>(&self, record: BertIntermediateRecord<B>) -> BertIntermediate<B> {
      let dense = LinearConfig::new(self.hidden_size, self.intermediate_size).init_with(record.dense);
      let intermediate_act = GELU::new();

      BertIntermediate { 
        dense,
        intermediate_act,
      }
    }
}

impl<B: Backend> BertIntermediate<B> {
  pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
    let hidden_states = self.dense.forward(hidden_states);
    self.intermediate_act.forward(hidden_states)
  }
}

// BertOutput Configuration and Module
#[derive(Config, Debug)]
pub struct BertOutputConfig {
  pub intermediate_size: usize,
  pub hidden_size: usize,
  pub layer_norm_eps: f64,
  pub hidden_dropout_prob: f64,
}

#[derive(Module, Debug)]
pub struct BertOutput<B: Backend> {
  dense: Linear<B>,
  layer_norm: LayerNorm<B>,
  dropout: Dropout,
}

impl BertOutputConfig {
    pub fn init<B: Backend>(&self) -> BertOutput<B> {
      let dense = LinearConfig::new(self.intermediate_size, self.hidden_size).init();
      let layer_norm_config = LayerNormConfig::new(self.hidden_size);
      let layer_norm_config = layer_norm_config.with_epsilon(self.layer_norm_eps);
      let layer_norm = layer_norm_config.init();

      let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

      BertOutput {
          dense,
          layer_norm,
          dropout,
      }
    }

    pub fn init_with<B: Backend>(&self, record: BertOutputRecord<B>) -> BertOutput<B> {
      let dense = LinearConfig::new(self.intermediate_size, self.hidden_size).init_with(record.dense);
      let layer_norm_config = LayerNormConfig::new(self.hidden_size);
      let layer_norm_config = layer_norm_config.with_epsilon(self.layer_norm_eps);
      let layer_norm = layer_norm_config.init_with(record.layer_norm);
      
      let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

      BertOutput {
          dense,
          layer_norm,
          dropout,
      }
    }
}

impl<B: Backend> BertOutput<B> {
  pub fn forward(&self, hidden_states: Tensor<B, 3>, input_tensor: Tensor<B, 3>) -> Tensor<B, 3> {
    let hidden_states = self.dense.forward(hidden_states);
    let hidden_states = self.dropout.forward(hidden_states);
    let result = self.layer_norm.forward(hidden_states + input_tensor);
    result
  }
}


// Configuration for BertSelfAttention
#[derive(Config, Debug)]
pub struct BertSelfAttentionConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub attention_head_size: usize,
    pub hidden_dropout_prob: f64,
}

#[derive(Module, Debug)]
pub struct BertSelfAttention<B: Backend> {
    pub query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
}

impl BertSelfAttentionConfig {
    pub fn init<B: Backend>(&self) -> BertSelfAttention<B> {
        let all_head_size = self.num_attention_heads * self.attention_head_size;
        let query = LinearConfig::new(self.hidden_size, all_head_size).init();
        let key = LinearConfig::new(self.hidden_size, all_head_size).init();
        let value = LinearConfig::new(self.hidden_size, all_head_size).init();
        let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

        BertSelfAttention {
            query,
            key,
            value,
            dropout,
            num_attention_heads: self.num_attention_heads,
            attention_head_size: self.attention_head_size,
            all_head_size,
        }
    }

    pub fn init_with<B: Backend>(&self, record: BertSelfAttentionRecord<B>) -> BertSelfAttention<B> {
        let all_head_size = self.num_attention_heads * self.attention_head_size;
        let query = LinearConfig::new(self.hidden_size, all_head_size).init_with(record.query);
        let key = LinearConfig::new(self.hidden_size, all_head_size).init_with(record.key);
        let value = LinearConfig::new(self.hidden_size, all_head_size).init_with(record.value);
        let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

        BertSelfAttention {
            query,
            key,
            value,
            dropout,
            num_attention_heads: self.num_attention_heads,
            attention_head_size: self.attention_head_size,
            all_head_size,
        }
    }
}

impl<B: Backend> BertSelfAttention<B> {
  pub fn forward(&self, hidden_states: Tensor<B, 3>, mask_attn: Tensor<B, 4>) -> Tensor<B, 3> {
    let query_layer = self.query.forward(hidden_states.clone());
    let key_layer = self.key.forward(hidden_states.clone());
    let value_layer = self.value.forward(hidden_states.clone());

    // println!("Key pre transpose {:?}", &key_layer.clone().to_data().value[30..60]);

    let query_layer = self.transpose_for_scores(&query_layer);  // I'm using unwrap here for brevity.
    let key_layer = self.transpose_for_scores(&key_layer);
    let value_layer = self.transpose_for_scores(&value_layer);

    // Print the shape of the query, key, and value layers
    // println!("Query {:?}", &query_layer.clone().to_data().value[30..60]);
    // println!("Key {:?}", &key_layer.clone().to_data().value[30..60]);
    // println!("Value {:?}", &value_layer.clone().to_data().value[30..60]);

    // println!("Key layer transposed {:?}", &key_layer.clone().swap_dims(2, 3).to_data().value[30..60]);
    // println!("Key layer transposed shape {:?}", &key_layer.clone().swap_dims(2, 3).shape());

    let attention_scores = query_layer.clone().matmul(key_layer.swap_dims(2, 3)).div_scalar(sqrtf(self.attention_head_size as f32));
    // println!("Attention Scores {:?}", &attention_scores.clone().to_data().value[0..32]);
    // println!("Mask Attn {:?}", &mask_attn.clone().to_data().value[0..8]);
    let attention_scores = attention_scores + mask_attn;
    let attention_probs = activation::softmax(attention_scores, 3);
    let attention_probs = self.dropout.forward(attention_probs);

    let context_layer = attention_probs.matmul(value_layer);

    // Print the shape of the context layer
    // println!("Context Layer {:?}", &context_layer.clone().to_data().value[0..8]);
    let context_layer = context_layer.swap_dims(1, 2).flatten(2, 3);
    // println!("Context Layer Reshaped {:?}", &context_layer.clone().to_data().value[0..8]);
    context_layer
  }

  fn transpose_for_scores(&self, xs: &Tensor<B, 3>) -> Tensor<B, 4> {
    let mut new_x_shape = xs.dims().to_vec();
    new_x_shape.pop();
    new_x_shape.push(self.num_attention_heads);
    new_x_shape.push(self.attention_head_size);

    // Convert vector to an array of size 4
    let array_shape: [usize; 4] = match new_x_shape.as_slice() {
      &[a, b, c, d] => [a, b, c, d],
      c => panic!("Unexpected tensor dimensions {:?}", c),
    };

    // println!("Array Shape {:?}", array_shape.clone());
    let xs = xs.clone().reshape(array_shape).swap_dims(1, 2);
    xs
  }
}

// Configuration for BertSelfOutput
#[derive(Config, Debug)]
pub struct BertSelfOutputConfig {
    pub hidden_size: usize,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
}

#[derive(Module, Debug)]
pub struct BertSelfOutput<B: Backend> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

impl BertSelfOutputConfig {
  pub fn init<B: Backend>(&self) -> BertSelfOutput<B> {
    let dense = LinearConfig::new(self.hidden_size, self.hidden_size).init();
    let layer_norm_config = LayerNormConfig::new(self.hidden_size);
    let layer_norm_config = layer_norm_config.with_epsilon(self.layer_norm_eps);
    let layer_norm = layer_norm_config.init();

    let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

    BertSelfOutput {
        dense,
        layer_norm,
        dropout,
    }
  }

  pub fn init_with<B: Backend>(&self, record: BertSelfOutputRecord<B>) -> BertSelfOutput<B> {
    let dense = LinearConfig::new(self.hidden_size, self.hidden_size).init_with(record.dense);
    let layer_norm_config = LayerNormConfig::new(self.hidden_size);
    let layer_norm_config = layer_norm_config.with_epsilon(self.layer_norm_eps);
    let layer_norm = layer_norm_config.init_with(record.layer_norm);
    
    let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

    BertSelfOutput {
        dense,
        layer_norm,
        dropout,
    }
  }
}

impl<B: Backend> BertSelfOutput<B> {
  pub fn forward(&self, hidden_states: Tensor<B, 3>, input_tensor: Tensor<B, 3>) -> Tensor<B, 3> {
    let hidden_states = self.dense.forward(hidden_states);
    let hidden_states = self.dropout.forward(hidden_states);
    let result = self.layer_norm.forward(hidden_states + input_tensor);
    result
  }
}
