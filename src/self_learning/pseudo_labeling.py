from transformers import Seq2SeqTrainer
from torch.nn.functional import softmax


class Seq2SeqTrainerForPseudoLabeling(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Label smoothing and weighting loss for pseudo-labeled data
        """
        labels = inputs.pop(
            "labels"
        )  # we assume that labels are presented as vectors regardless of whether they are smoothed
        weight = None
        if "weight" in inputs:
            weight = inputs.pop("weight")
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        logits = outputs.logits
        if labels.shape != logits.shape:
            labels = labels.unsqueeze(1)
            losses = -softmax(logits, dim=-1).log().gather(1, labels)
        else:
            losses = -softmax(logits, dim=-1).log() * labels

        if weight is not None:
            loss = (losses * weight).sum() / len(labels)
        else:
            loss = losses.sum() / len(labels)

        return (loss, outputs) if return_outputs else loss
