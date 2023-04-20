from transformers import Seq2SeqTrainingArguments


def get_training_args():
    args = Seq2SeqTrainingArguments(
        output_dir="tmp/model_output",
        evaluation_strategy="epoch",
        learning_rate=5e-6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
    )
    return args
