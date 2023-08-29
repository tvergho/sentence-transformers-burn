![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)
[![Rust Version](https://img.shields.io/badge/Rust-1.65.0+-blue)](https://releases.rs/docs/1.65.0)

# Sentence Transformers in Burn (ST-Burn)

This library provides an implementation of the [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) framework for computing text representations as vector embeddings in Rust. Specifically, it uses the [Burn](https://github.com/burn-rs/burn) deep learning to implement the **BERT model**. Using Burn, this can be combined with any supported backend for fast, efficient, cross-platform inference on CPUs and GPUs. ST-Burn supports any [state-of-the-art model](https://huggingface.co/spaces/mteb/leaderboard) that implements the BERT architecture.

Currently **inference-only** for now.

## Features
- Import models via `safetensors` (using [Candle](https://github.com/huggingface/candle)) 📦
- Code structure replicates the official Huggingface `BertModel` implementation 🚀
- Flexible inference backend using Burn 🔧

## Installation
`sentence-transformers-burn` can be installed from source.

```
cargo add --git https://github.com/tvergho/sentence-transformers-burn.git sentence_transformers
```

Run `cargo build` to make sure everything can be correctly built.

```
cargo build
```

Note that building the `burn-tch` dependency may require manually linking Libtorch. After installing via pip:

```
export LIBTORCH=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
# /path/to/torch

export DYLD_LIBRARY_PATH=/path/to/torch/lib
```

Python dependencies (for running the scripts in `scripts/`) should also be installed.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
A `BertModel` can be loaded and initialized from a file, as in the example below:

```rs
use sentence_transformers::bert_loader::{load_model_from_safetensors, load_config_from_json};
use sentence_transformers::model::{
  bert_embeddings::BertEmbeddingsInferenceBatch,
  bert_model::BertModel,
};
use burn_tch::{TchBackend, TchDevice};
use burn::tensor::Tensor;

const BATCH_SIZE: u64 = 64;

let device = TchDevice::Cpu;
let config = load_config_from_json("model/bert_config.json");
let model: BertModel<_> = load_model_from_safetensors::<TchBackend<f32>>("model/bert_model.safetensors", &device, config);

let batch = BertEmbeddingsInferenceBatch {
  tokens: Tensor::zeros(vec![BATCH_SIZE, 256]).to_device(&device.clone()),
  mask_attn: Some(Tensor::ones(vec![BATCH_SIZE, 256]).to_device(&device.clone()))
};

model.forward(batch); // [batch_size, seq_len, n_dims]
```

`sentence-transformers-burn` also comes with a built-in **inference server**. To start, simply run:

```
cargo run --bin server -- path/to/model/dir
```

The model directory should contain a `bert_model.safetensors` and `bert_config.json` file. Once the server is running, inference can be initiated via POST request:

```json
POST http://localhost:3030/embed

{
  "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
	"attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
}
```

This will return a 3D array of floats of size `[batch_size, seq_len, n_dims]`. 

## Testing
Tests can be run to verify that the Rust model output matches a comparable Huggingface model. To save a model to use during testing, run `python scripts/prepare_test.py`. Then, simply:

```
cargo run test
```