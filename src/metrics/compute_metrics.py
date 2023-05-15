from typing import Dict, List, Optional, Tuple, Union

import pickle

import numpy as np
from evaluate import load
from tqdm import tqdm

from .metrics import (
    calculate_abstractiveness_scores,
    decode,
    pair_bleu,
    exact_match_multiple,
)

SACREBLEU = load("sacrebleu")
ROUGE = load("rouge")
BERTSCORE = load("bertscore")


def compute_metrics(
    eval_preds,
    tokenizer,
    batch_size: int,
    add_metrics_to_use: Union[Tuple[str, ...], List[str]] = (
        "bartscore",
        "bertscore",
        "sentbert",
        "summac",
        "cola",
    ),
    aggregate_cola: bool = False,
) -> Dict[str, float]:
    generated_texts, reference_texts, *original_texts = decode(eval_preds, tokenizer)
    if len(original_texts) > 0:
        original_texts = original_texts[0]

    result = {}
    ### Metrics than only employ the generated texts
    result["word_length_gen"] = np.array(
        [len(text.split()) for text in generated_texts]
    )
    result["char_length_gen"] = np.array([len(text) for text in generated_texts])
    result["token_length_gen"] = np.array(
        [
            np.count_nonzero(pred != tokenizer.pad_token_id) - 3
            for pred in eval_preds.predictions
        ]
    )

    ### Metrics that use both the generated texts and the original texts and
    ### those than are not ''obliged'' to use reference texts
    # Lengths
    src_word_lengths = [len(text.split()) for text in original_texts]
    src_char_lengths = [len(text) for text in original_texts]
    result["word_length_src_rel"] = np.mean(
        result["word_length_gen"] / src_word_lengths
    )
    result["char_length_src_rel"] = np.mean(
        result["char_length_gen"] / src_char_lengths
    )

    if "cola" in add_metrics_to_use:
        from .metrics import calculate_cola_model_predictions
        
        with open("texts.pkl", "wb") as fd:
            pickle.dump(generated_texts, fd)
        result["cola_score"] = calculate_cola_model_predictions(
            generated_texts,
            aggregate=aggregate_cola,
            batch_size=batch_size
        )
        # Relative cola score
        src_cola_score = calculate_cola_model_predictions(
            original_texts,
            aggregate=aggregate_cola,
            batch_size=batch_size,
        )
        result["cola_score_src_rel"] = result["cola_score"] / src_cola_score
        # Relative cola score
        ref_cola_score = calculate_cola_model_predictions(
            reference_texts,
            aggregate=aggregate_cola,
            batch_size=batch_size
        )
        result["cola_score_rel"] = result["cola_score"] / ref_cola_score

    if "bertscore" in add_metrics_to_use:
        bertscore_art = BERTSCORE.compute(
            predictions=generated_texts,
            references=original_texts,
            model_type="roberta-large",
        )
        for key in bertscore_art:
            if key != "hashcode":
                result[f"bertscore_art_{key}"] = bertscore_art[key]
    result.update(
        calculate_abstractiveness_scores(
            generated_texts, original_texts, reference_texts
        )
    )
    if "summac" in add_metrics_to_use:
        from .metrics import calculate_summac_score
        result.update(
            calculate_summac_score(generated_texts, original_texts, reference_texts)
        )
    if "bartscore" in add_metrics_to_use:
        from .metrics import calculate_bart_score
        result.update(
            calculate_bart_score(
                preds=generated_texts,
                texts=original_texts,
                refs=reference_texts,
                batch_size=batch_size,
            )
        )
    if "sentbert" in add_metrics_to_use:
        from .metrics import SentBert
        sentbert = SentBert()
        result["sentbert_src"] = sentbert(
            original_texts, generated_texts, batch_size=batch_size
        )
    ### Metrics that use both the generated texts and the reference texts
    if reference_texts is not None:
        # BLEU
        result["bleu"] = np.array(
            [
                pair_bleu(pred, ref)
                for pred, ref in tqdm(zip(generated_texts, reference_texts))
            ]
        )
        # ROUGE
        result.update(
            ROUGE.compute(
                predictions=generated_texts,
                references=reference_texts,
                use_stemmer=True,
            )
        )
        # Sacrebleu
        sacrebleu_result = SACREBLEU.compute(
            predictions=generated_texts, references=[[ref] for ref in reference_texts]
        )
        result["sacrebleu"] = sacrebleu_result.pop("score")
        # Lengths
        ref_word_lengths = [len(text.split()) for text in reference_texts]
        ref_char_lengths = [len(text) for text in reference_texts]
        ref_token_lengths = [
            np.count_nonzero(lab != tokenizer.pad_token_id) - 2
            for lab in eval_preds.label_ids
        ]
        result["word_length_rel"] = result["word_length_gen"] / ref_word_lengths
        result["char_length_rel"] = result["char_length_gen"] / ref_char_lengths
        result["token_length_rel"] = result["token_length_gen"] / ref_token_lengths
        # BERTScore
        if "bertscore" in add_metrics_to_use:
            bertscore = BERTSCORE.compute(
                predictions=generated_texts,
                references=reference_texts,
                model_type="roberta-large",
            )
            for key in bertscore:
                if key != "hashcode":
                    result[f"bertscore_{key}"] = bertscore[key]

        if "sentbert" in add_metrics_to_use:
            from .metrics import SentBert
            sentbert = SentBert()
            result["sentbert_ref"] = sentbert(reference_texts, generated_texts, batch_size=batch_size)
            sentbert_ref_src = sentbert(original_texts, reference_texts, batch_size=batch_size)
            result["sentbert_rel"] = result["sentbert_src"] / sentbert_ref_src

    for key, value in result.items():
        if not isinstance(value, (int, float)):
            result[key] = float(np.mean(value))

    return result


def compute_metrics_for_qa(eval_preds, tokenizer, answers_by_question: dict):
    generated_texts, reference_texts, *original_texts = decode(eval_preds, tokenizer)
    if len(original_texts) > 0:
        original_texts = original_texts[0]
    else:
        raise Exception("Original texts have not been passed to metrics computation. Use include_inputs_for_metrics=True in training args")

    references_with_aliases = [answers_by_question[question] for question in original_texts]

    result = {
        "exact_match": exact_match_multiple(predictions=generated_texts, references=references_with_aliases)
    }
    return result
