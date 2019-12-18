from typing import List, Dict

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('slam_predictor')
@Predictor.register('slam-predictor')
class LiePredictor(Predictor):
    pass
