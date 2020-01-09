import torch
import torch.nn.functional as F


def weighted_sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                                  targets: torch.LongTensor,
                                                  mask: torch.FloatTensor,
                                                  weights: torch.FloatTensor = None) -> torch.FloatTensor:
    num_classes = logits.size(-1)
    # make sure weights are float
    mask = mask.float().view(-1)
    logits = logits.view(-1, num_classes)
    targets = targets.view(-1)
    #print(mask.shape, logits.shape, targets.shape)
    loss = F.cross_entropy(logits, targets, weight=weights, reduction='none')
    #print(loss.shape)
    loss = (loss * mask).mean()
    #print('')
    return loss
