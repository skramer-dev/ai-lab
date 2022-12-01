from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, EarlyStoppingCallback
import numpy as np
from datasets import load_metric

from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast
)


def compute_metrics(eval_pred):
    """ Computes the metrics given a tuple of (logits, labels) """

    load_accuracy = load_metric("accuracy")
    precision_metric = load_metric('precision')
    recall_metric = load_metric('recall')
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(
        predictions=predictions, references=labels)["accuracy"]
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="macro", zero_division=0)["precision"]
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="macro", zero_division=0)["recall"]
    f1 = load_f1.compute(predictions=predictions,
                         references=labels, average="macro")["f1"]

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# load data
pokemon_descriptions = load_dataset('./', delimiter=';')

NUM_CLASSES = len(np.unique(pokemon_descriptions['train']['labels']))

# train test split
split_pokemon_descriptions = pokemon_descriptions['train'].train_test_split(
    test_size=0.2, shuffle=True)

# load tokenizer
tokenizer_bert = AutoTokenizer.from_pretrained("bert-large-uncased")

# tokenize data
tokenized_pokemon_descriptions = split_pokemon_descriptions.map(
    lambda example: tokenizer_bert(example['text'], truncation=True, padding=True), batched=True)

# remove unwanted columns in data
tokenized_pokemon_descriptions['train'] = tokenized_pokemon_descriptions['train'].remove_columns([
                                                                                                 'text', 'name'])
tokenized_pokemon_descriptions['test'] = tokenized_pokemon_descriptions['test'].remove_columns([
                                                                                               'text', 'name'])

def setup_trainer_for_finetuning(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast) -> Trainer:

    finetuning_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=40,
        weight_decay=0.01,
        warmup_ratio=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        seed=42,
        dataloader_num_workers=6,
        metric_for_best_model = 'f1',
        load_best_model_at_end=True
    )

    return Trainer(
        model=model,
        args=finetuning_args,
        train_dataset=tokenized_pokemon_descriptions['train'],
        eval_dataset=tokenized_pokemon_descriptions['test'],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )


# load model
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-large-uncased", num_labels=NUM_CLASSES)

# setup trainer
trainer = setup_trainer_for_finetuning(
    model=bert_model, tokenizer=tokenizer_bert)

# run training loop
trainer.train()

# save model
trainer.save_model('./saved-model/')
