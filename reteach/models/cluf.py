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

from ..util import weighted_sequence_cross_entropy_with_logits
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
                 user_feature_embedder: TextFieldEmbedder,
                 format_feature_embedder: TextFieldEmbedder,
                 embedder: TextFieldEmbedder,
                 context_encoder_word: Seq2SeqEncoder,
                 context_encoder_character: Seq2VecEncoder,
                 user_encoder: FeedForward,
                 format_encoder: FeedForward,
                 global_encoder: FeedForward,
                 local_encoder: FeedForward,
                 classifier: FeedForward,
                 linguistic_encoder: Seq2SeqEncoder = None,
                 alpha: float = 0.7,
                 dropout: float = 0.5) -> None:
        super().__init__(vocab)

        # The idea behind the CLUF model is to separately encode the:
        #  (C)ontext, (L)inguistic Features, (U)ser, and (F)ormat
        # which is what the following parameters do.

        # (C)ontext Encoder
        self._context_encoder_word = context_encoder_word
        self._context_encoder_character = context_encoder_character
        # (L)inguistic Encoder
        self._linguistic_encoder = linguistic_encoder
        # (U)ser Encoder
        self._user_encoder = user_encoder
        # (F)ormat Encoder
        self._format_encoder = format_encoder

        # projection for global/local information
        self._global_encoder = global_encoder
        self._local_encoder = TimeDistributed(local_encoder)

        self._classifier = TimeDistributed(classifier)
        self._embedder = embedder
        self._user_feature_embedder = user_feature_embedder
        self._format_feature_embedder = format_feature_embedder

        self._dropout = nn.Dropout(dropout)
        self._auc = Auc()
        self._f1 = F1Measure(1)
        self._alpha = alpha

        self._user_features = ['user', 'countries']
        self._format_features = ['client', 'session', 'format']

    def forward(self, words: Dict[str, torch.Tensor],
                days: torch.Tensor, time: torch.Tensor,
                labels: torch.Tensor, metadata: Optional[Any] = None,
                **features) -> Dict[str, torch.Tensor]:
        # shape : (batch, features)
        features = reduce(lambda x,y: dict(chain(x.items(), y.items())), features.values(), {})
        # shape : (batch, features)
        feature_mask = get_text_field_mask(features)

        # split the features into user features and format features
        user_features = { k: v for k, v in features.items() if k in self._user_features }
        format_features = { k: v for k, v in features.items() if k in self._format_features }

        # shape: (batch, 1, user_features)
        user_feature_values = self._user_feature_embedder(user_features)
        # shape: (batch, 1, format_features)
        format_feature_values = self._format_feature_embedder(format_features)

        # shape : (batch, words)
        mask = get_text_field_mask(words)
        #
        b, num_words = mask.shape
        # shape : (batch, words, embeddings)
        words = self._embedder(words)
        words = self._dropout(words)
        # shape : (batch, words, embeddings)
        words = self._context_encoder_word(words, mask)
        words = self._dropout(words)
        # shape : (batch, words, local_dim)
        local_information = self._local_encoder(words)

        # shape : (batch, users_dim)
        users_encoded = self._user_encoder(torch.cat([user_feature_values.squeeze(1), days], dim=1))
        # shape : (batch, format_dim)
        format_encoded = self._format_encoder(torch.cat([format_feature_values.squeeze(1), time], dim=1))
        # shape : (batch, users_dim+format_dim)
        global_information = torch.cat([users_encoded, format_encoded], dim=1)
        # shape : (batch, global_dim)
        global_information = self._global_encoder(global_information)
        # shape : (batch, words, global_dim)
        global_information = global_information.unsqueeze(1)
        global_information = global_information.repeat(1, num_words, 1)

        # shape : (batch, words, global/local_dim)
        mixed = local_information * global_information
        # shape : (batch, words, 2)
        logits = self._classifier(mixed)

        output: Dict[str, torch.Tensor] = {}
        output['logits'] = logits

        if labels is not None:
            output['loss'] = weighted_sequence_cross_entropy_with_logits(logits, labels, mask, weights = torch.Tensor([1. - self._alpha, self._alpha]))
            # output['loss'] = sequence_cross_entropy_with_logits(logits, labels, mask)
            # shape : (batch * words,)
            logits = F.softmax(logits, dim=-1)
            logits = logits.view(b * num_words, -1)
            # shape : (batch * words,)
            auc_mask = mask.view(b * num_words)
            # shape : (batch * words)
            labels = labels.view(b * num_words)

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
