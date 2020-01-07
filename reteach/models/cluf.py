from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.auc import Auc
from allennlp.training.metrics.f1_measure import F1Measure

from typing import Dict, Optional, Any
from itertools import chain
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


@Model.register('cluf')
class CLUFModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 classifier: FeedForward,
                 user_encoder: FeedForward,
                 format_encoder: FeedForward,
                 feature_embedder: TextFieldEmbedder,
                 embedder: TextFieldEmbedder,
                 word_context_encoder: Optional[Seq2SeqEncoder] = None,
                 character_context_encoder: Optional[Seq2VecEncoder] = None,
                 linguistic_encoder: Optional[Seq2SeqEncoder] = None,
                 dropout: float = 0.5) -> None:
        super().__init__(vocab)

        # The idea behind the CLUF model is to separately encode the:
        #  (C)ontext, (L)inguistic Features, (U)ser, and (F)ormat
        # which is what the following parameters do.

        # (C)ontext Encoder
        self._word_context_encoder = word_context_encoder
        self._character_context_encoder = character_context_encoder
        # (L)inguistic Encoder
        self._linguistic_encoder = linguistic_encoder
        # (U)ser Encoder
        self._user_encoder = user_encoder
        # (F)ormat Encoder
        self._format_encoder = format_encoder

        self._classifier = TimeDistributed(classifier)
        self._embedder = embedder
        self._feature_embedder = feature_embedder
        self._encoder = encoder
        self._dropout = nn.Dropout(dropout)
        self._auc = Auc()
        self._f1 = F1Measure(1)
        #self._pos_weight = 2.0

    def forward(self, words: Dict[str, torch.Tensor],
                days: torch.Tensor, time: torch.Tensor,
                labels: torch.Tensor, metadata: Optional[Any] = None,
                **features) -> Dict[str, torch.Tensor]:
        # shape : (batch, features)
        features = reduce(lambda x,y: dict(chain(x.items(), y.items())), features.values(), {})
        # shape : (batch, features)
        feature_mask = get_text_field_mask(features)
        feature_values = self._feature_embedder(features)
        b, _, f = feature_values.shape
        # shape : (batch, words)
        mask = get_text_field_mask(words)
        #
        batch, num_words = mask.shape
        # shape : (batch, words, embeddings)
        words = self._embedder(words)
        words = self._dropout(words)
        words = torch.cat([words, feature_values.expand(b, num_words, f), days.unsqueeze(1).expand(b, num_words, 1), time.unsqueeze(1).expand(b, num_words, 1)], dim=2)
        # shape : (batch, words, embeddings)
        words = self._encoder(words, mask)
        words = self._dropout(words)
        # shape : (batch, words, classes)
        logits = self._classifier(words)

        output: Dict[str, torch.Tensor] = {}
        output['logits'] = logits

        if labels is not None:
            # output['loss'] = weighted_sequence_cross_entropy_with_logits(logits, labels, mask, weights = torch.Tensor([1.0, self._pos_weight]))
            output['loss'] = sequence_cross_entropy_with_logits(logits, labels, mask)
            # shape : (batch * words,)
            logits = F.softmax(logits, dim=-1)
            logits = logits.view(batch * num_words, -1)
            # shape : (batch * words,)
            auc_mask = mask.view(batch * num_words)
            # shape : (batch * words)
            labels = labels.view(batch * num_words)

            self._auc(logits[:, 1], labels, auc_mask)
            self._f1(logits, labels, auc_mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'auc': self._auc.get_metric(reset)
        }
        #print(self._f1.get_metric(reset))
        metrics.update(dict(zip(['p', 'r', 'f1'], self._f1.get_metric(reset))))
        return metrics
