from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_from_disk
import evaluate
import numpy as np

import config

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)
  return accuracy.compute(predictions=predictions, references=labels)

def evaluate_model():
    model = AutoModelForImageClassification.from_pretrained(config.BEST_MODEL_CHECKPOINT)

    eval_args = TrainingArguments(**config.EVAL_ARGS)

    trainer = Trainer(
        model = model,
        args = eval_args,
        compute_metrics=compute_metrics
    )

    eval_dataset = load_from_disk(config.PROCESSED_TEST_DATA)

    predictions = trainer.evaluate(eval_dataset=eval_dataset)

    eval_accuracy = f"{predictions['eval_accuracy']:.4f}"
    eval_loss = f"{predictions['eval_loss']:.4f}"

    return print(f"Eval accuracy: {eval_accuracy}, eval loss: {eval_loss}")


if __name__ == "__main__":
    evaluate_model()