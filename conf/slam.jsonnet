local cuda = std.parseInt(std.extVar('cuda'));
local pair = "fr_en";

local base = if (cuda == 0) then "/home/jdbarrow/Research/data/slam/" + pair + "/" else "/fs/clip-scratch-new/joe/slam/" + pair + "/";

local features = ['user', 'countries', 'client', 'session', 'format'];

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
        for feature in features
      }
   },
   iterator: {
      batch_size: 32,
      type: "basic"
//      type: "bucket",
//      sorting_keys: [["tokens", "num_tokens"]]
   },
   model: {
     type: "user_model_lstm",
     embedder: {
       allow_unmatched_keys: true,
       tokens: {
         type: "embedding",
//         pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
        pretrained_file: "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz",
         embedding_dim: 300,
         trainable: true
       }
     },
     feature_embedder: {
       [feature]: {
         type: "embedding",
         embedding_dim: 10,
         trainable: true
       } for feature in features
     } + { allow_unmatched_keys: true },
     encoder: {
       type: 'lstm',
       input_size: 352,
       hidden_size: 300,
       bidirectional: true,
       dropout: 0.1
     },
     classifier: {
       input_dim: 600,
       num_layers: 1,
       hidden_dims: 2,
       activations: 'linear'
     },
     dropout: 0.3
   },
   trainer: {
     num_epochs: 10,
     patience: 5,
     cuda_device: cuda-1,
     grad_clipping: 5.0,
     validation_metric: '+auc',
     optimizer: {
       type: 'adam',
       lr: 0.003
     }
   }
}