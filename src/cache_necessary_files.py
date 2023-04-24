from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel
from transformers import AutoTokenizer

from datasets import load_dataset
from evaluate import load


MODELS_SEQ2SEQ = ["facebook/bart-base", "t5-base", "facebook/bart-large-cnn"]
MODELS_CLS = ["Aktsvigun/electra-large-cola"]
MODELS_GENERIC = ["sentence-transformers/all-mpnet-base-v2"]
DATASETS = ["xsum", "aeslc", ("trivia_qa", "unfiltered.nocontext"), "eli5", "quora"]
METRICS = ["sacrebleu", "rouge", "bertscore"]


def main():
    for model in MODELS_SEQ2SEQ:
        AutoModelForSeq2SeqLM.from_pretrained(model)
        AutoTokenizer.from_pretrained(model)
    for model in MODELS_CLS:
        AutoModelForSequenceClassification.from_pretrained(model)
        AutoTokenizer.from_pretrained(model)
    for model in MODELS_GENERIC:
        AutoModel.from_pretrained(model)
        AutoTokenizer.from_pretrained(model)

    for dataset in DATASETS:
        if isinstance(dataset, tuple):
            dataset, subset = dataset
        else:
            subset = None
        load_dataset(dataset, subset)

    for metric in METRICS:
        load(metric)


if __name__ == "__main__":
    main()
