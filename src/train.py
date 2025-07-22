from transformers import AutoModelForImageClassification, AutoImageProcessor, DefaultDataCollator, TrainingArguments, Trainer
import numpy as np
import evaluate
from datasets import load_from_disk

import config
from data_processing import labels, id2label, label2id

model = AutoModelForImageClassification.from_pretrained(config.MODEL_CHECKPOINT, num_labels=len(labels), id2label=id2label, label2id=label2id)
processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=True)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)
  return accuracy.compute(predictions=predictions, references=labels)

data_collator = DefaultDataCollator()

training_args = TrainingArguments(**config.TrainingArgs)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator=data_collator,
    train_dataset = load_from_disk(config.PROCESSED_TRAIN_DATA),
    eval_dataset = load_from_disk(config.PROCESSED_TEST_DATA),
    tokenizer = processor,
    compute_metrics = compute_metrics
)

if __name__ == "__main__":
  trainer.train()