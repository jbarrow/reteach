from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('slam_predictor')
@Predictor.register('slam-predictor')
class LiePredictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        outputs['words'] = [str(token) for token in instance.fields['words'].tokens]
        outputs['predicted'] = [l for l in outputs['logits'].argmax(1)]
        outputs['labels'] = instance.fields['labels'].labels

        for field in ['user', 'client', 'format', 'countries', 'session']:
            outputs[field] = instance.fields[field].tokens[0]

        for field in ['days', 'time']:
            outputs[field] = instance.fields[field].array[0]
            #print(field, instance.fields[field], dir(instance.fields[field]))

        return sanitize(outputs)
