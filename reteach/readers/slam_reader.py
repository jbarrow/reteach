from typing import Dict, Tuple, List, Iterable
import logging

from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

FIELDS = [
    'form',
    'upostag',
    'feats',
    'deprel',
    'head'
]

def unpack_token_index(token_index: str) -> Tuple[str, int, int]:
    """
    The Duolingo SLAM dataset packs the following information into the
    token index:

        [_ _ _ _ _ _ _ _] [_ _] [_ _]

    The first 8 digits are the base64 encoded session ID, the next 2 are
    the index of the exercise within the session, and the final 2 are the
    index of the token in the exercise.
    """
    return {
        'session_id': token_index[:8],
        'exercise_id': int(token_index[8:10]),
        'id': int(token_index[10:])
    }


def lazy_parse(text: str, fields: Tuple[str, ...]=DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if not sentence: continue
        annotation = []
        features = {}
        for line in sentence.split("\n"):
            if line.strip().startswith("#"):
                if line[:8] == '# prompt':
                    features['prompt'] = line.strip().split(':')[1]
                else:
                    new_features = line.strip()[1:].split()
                    for new_feature in new_features:
                        name, value = new_feature.split(':')
                        features[name] = value
                continue

            index_label, *data = line.strip().split()
            *data, label = data

            output = parse_line('\t'.join(data), fields)
            output.update(unpack_token_index(index_label))
            output.update({'label': int(label)})

            annotation.append(output)

        yield annotation, features


@DatasetReader.register("slam_reader")
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
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as conllu_file:
            logger.info("Reading token instances from conllu dataset at: %s", file_path)

            for annotation, features in lazy_parse(conllu_file.read(), fields=FIELDS):
                annotation = [x for x in annotation if x["id"] is not None]

                labels = [x["label"] for x in annotation]
                words = [x["form"] for x in annotation]

                yield self.text_to_instance(words, labels)

    @overrides
    def text_to_instance(self,  # type: ignore
                         words: List[str],
                         token_labels: List[int] = None) -> Instance:
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

        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["words"] = tokens
        if token_labels is not None:
            fields["labels"] = SequenceLabelField(token_labels, tokens,
                                                     label_namespace="token_labels")

        fields["metadata"] = MetadataField({"words": words})
        return Instance(fields)
