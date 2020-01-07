# reteach

`reteach` is a new approach to deep knowledge tracing by considering:

1. the timeline of the user studying each individual knowledge concept; and
2. the interaction between different concepts.

By taking this approach, we hope to build o

## Getting Started

### Installation

To install `reteach`, you must first install the requirements:

```
pip install -r requirements.txt
```

Then, just clone the repo and you should be good to go.

### Data

The original data can be found on the [Duolingo 2018 SLAM Shared Task page](https://sharedtask.duolingo.com/2018.html).
The validation and test datasets were blind and thus came with labels in a separate file.
To join the labels with the data, just run our `join_labels.py` script:

```
python -m scripts.join_labels [PATH_TO_DATA_FILE] [PATH_TO_KEY_FILE] > [PATH_TO_OUTPUT_FILE]
```

## Training

To train a model, run:

```
cuda=0 allennlp train \
  -f \
  --include-package reteach \
  -s [TRAINED_MODEL_DIRECTORY] \
  conf/slam.jsonnet
```

## Inference

You can generate a prediction file (with JSON outputs) by running the following command:

```
allennlp predict \
  --output-file [OUTPUT_FILE] \
  --silent \
  --include-package reteach \
  --predictor slam-predictor \
  --use-dataset-reader \
  [TRAINED_MODEL_DIRECTORY] \
  [DATASET]
```
