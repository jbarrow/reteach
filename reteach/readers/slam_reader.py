from typing import Dict, Tuple, List, Iterable
from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from .baseline_slam_reader import unpack_token_index, lazy_parse, FIELDS

import logging
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

token_features = ['Gender', 'Person', 'VerbForm', 'Definite', 'Mood', 'PronType', 'fPOS', 'Number', 'Tense']

@DatasetReader.register("slam_reader")
@DatasetReader.register("slam-reader")
class SLAMDatasetReader(DatasetReader):
    """
    Reads in a CoNLL-U formatted SLAM dataset, including all Duolingo-specific
    and UD-specific features.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 include_pos_features: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._include_pos_features = include_pos_features

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        # i = 0

        with open(file_path, 'r') as conllu_file:
            logger.info("Reading token instances from conllu dataset at: %s", file_path)

            for annotation, features in lazy_parse(conllu_file.read(), fields=FIELDS):
                # i += 1
                # if i == 10000: break

                annotation = [x for x in annotation if x["id"] is not None]

                labels = [x["label"] for x in annotation]
                words = [x["form"] for x in annotation]

                token_level = [{k: v for k, v in x['feats'].items() if k in token_features} for x in annotation]

                numerical = {}
                categorical = {}
                token_level = {}

                for k, v in features.items():
                    if k in ['days', 'time']:
                        try:
                            numerical[k] = min(float(v), 100.)
                        except ValueError:
                            # TODO: do some smarter missing value imputation
                            numerical[k] = 0.
                    else:
                        categorical[k] = v

                yield self.text_to_instance(words, labels, categorical, numerical, token_level)

    @overrides
    def text_to_instance(self,  # type: ignore
                         words: List[str],
                         token_labels: List[int] = None,
                         categorical: Dict[str, str] = {},
                         numerical: Dict[str, float] = {},
                         token_level: Dict[str, List[str]] = {}) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        token_labels : ``List[int]``, optional.
            The label for whether or not each token is correct.

        Returns
        -------
        An instance containing words and token labels.
        """
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(w) for w in words], {"tokens": self._token_indexers["tokens"]})
        fields["words"] = tokens

        for feature, value in categorical.items():
            if feature not in self._token_indexers.keys(): continue
            fields[feature] = TextField([Token(value)], { feature: self._token_indexers[feature] })

        for feature, value in numerical.items():
            fields[feature] = ArrayField(np.array([value]))

        #token_level_features = []
        #for token_features in token_level:
        #    fields[]

        if token_labels is not None:
            fields["labels"] = SequenceLabelField(token_labels, tokens,
                                                     label_namespace="token_labels")

        fields["metadata"] = MetadataField({"words": words})
        return Instance(fields)
