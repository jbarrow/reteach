from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.auc import Auc

from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

@Model.register('lstm_baseline')
class LSTMBaseline(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 classifier: FeedForward,
                 feature_embedder: TextFieldEmbedder,
                 embedder: TextFieldEmbedder) -> None:
        super().__init__(vocab)

        self._classifier = TimeDistributed(classifier)
        self._embedder = embedder
        self._feature_embedder = feature_embedder
        self._encoder = encoder
        self._auc = Auc()

    def forward(self, words: Dict[str, torch.Tensor],
                features: Dict[str, torch.Tensor],
                labels: torch.Tensor, metadata: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        # shape : (batch, features)
        feature_mask = get_text_field_mask(features)
        # shape : (batch, features)
        feature_values = self._feature_embedder(features)
        b, f = feature_values.shape
        # shape : (batch, words)
        mask = get_text_field_mask(words)
        #
        batch, num_words = mask.shape
        # shape : (batch, words, embeddings)
        words = self._embedder(words)
        words = torch.cat([words, feature_values[:, None, :].expand(b, num_words, f)], dim=2)
        # shape : (batch, words, embeddings)
        words = self._encoder(words, mask)
        # shape : (batch, words, classes)
        logits = self._classifier(words)

        output: Dict[str, torch.Tensor] = {}
        output['logits'] = logits

        if labels is not None:
            output['loss'] = sequence_cross_entropy_with_logits(logits, labels, mask)
            # shape : (batch * words,)
            logits = F.softmax(logits, dim=-1)
            logits = logits.view(batch * num_words, -1)
            logits = logits[:, 1]
            # shape : (batch * words,)
            auc_mask = mask.view(batch * num_words)
            # shape : (batch * words)
            labels = labels.view(batch * num_words)

            output['auc'] = self._auc(logits, labels, auc_mask)


        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'auc': self._auc.get_metric(reset)
        }
