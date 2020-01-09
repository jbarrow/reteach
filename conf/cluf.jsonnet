local cuda = std.parseInt(std.extVar('cuda'));
local pair = "fr_en";

local base = if (cuda == 0) then "/home/jdbarrow/Research/data/slam/" + pair + "/" else "/fs/clip-scratch-new/joe/slam/" + pair + "/";

local user_features = ['user', 'countries'];
local format_features = ['client', 'session', 'format'];

{
  train_data_path: base + pair + ".slam.20190204.train",
  validation_data_path: base + pair + ".slam.20190204.dev.joined",
  test_data_path: base + pair + ".slam.20190204.test.joined",
  evaluate_on_test: true,
   dataset_reader: {
      lazy: false,
      type: "slam_reader",
      token_indexers: {
        tokens: {
          type: "single_id",
          lowercase_tokens: true
        }
      } + {
        [feature]: { type: "single_id", namespace: feature }
        for feature in user_features
      } + {
        [feature]: { type: "single_id", namespace: feature }
        for feature in format_features
      }
   },
   iterator: {
      batch_size: 128,
      type: "basic"
   },
   model: {
     type: "cluf",
     embedder: {
       allow_unmatched_keys: true,
       tokens: {
         type: "embedding",
//        pretrained_file: "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz",
         embedding_dim: 300,
         trainable: true
       }
     },
     user_feature_embedder: {
       [feature]: {
         type: "embedding",
         embedding_dim: 100,
         trainable: true
       } for feature in user_features
     } + { allow_unmatched_keys: true },
     format_feature_embedder: {
       [feature]: {
         type: "embedding",
         embedding_dim: 100,
         trainable: true
       } for feature in format_features
     } + { allow_unmatched_keys: true },
     context_encoder_word: {
       type: 'lstm',
       input_size: 300,
       hidden_size: 100,
       num_layers: 2,
       bidirectional: true,
       dropout: 0.5
     },
     context_encoder_character: {
       type: 'lstm',
       input_size: 100,
       hidden_size: 100,
       num_layers: 3,
       bidirectional: true,
       dropout: 0.5
     },
     linguistic_encoder: {
       type: 'lstm',
       input_size: 200,
       hidden_size: 100,
       num_layers: 100,
       bidirectional: true
     },
     user_encoder: {
       input_dim: 201,
       num_layers: 1,
       hidden_dims: 200,
       activations: 'tanh'
     },
     format_encoder: {
       input_dim: 301,
       num_layers: 1,
       hidden_dims: 200,
       activations: 'tanh'
     },
     global_encoder: {
       input_dim: 400,
       hidden_dims: 200,
       num_layers: 1,
       activations: 'tanh'
     },
     local_encoder: {
       input_dim: 200,
       hidden_dims: 200,
       num_layers: 1,
       activations: 'tanh'
     },
     classifier: {
       input_dim: 200,
       num_layers: 1,
       hidden_dims: 2,
       activations: 'linear'
     },
     dropout: 0.5
   },
   trainer: {
     num_epochs: 10,
     patience: 5,
     cuda_device: cuda-1,
     grad_clipping: 5.0,
     validation_metric: '+f1',
     optimizer: {
       type: 'adam',
       lr: 0.001
     }
   }
}
