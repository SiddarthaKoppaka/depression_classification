import argparse
import pandas as pd
import numpy as np
import os
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.utils import shuffle
from scipy.special import softmax
from transformers import set_transformers_cache, logging
import torch

def main(args):
    # Set the GPU (if available)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # Configure transformers logging
    logging.set_verbosity_error()

    # Set up Hugging Face cache and authentication
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    os.environ['HF_HOME'] = args.cache_dir
    set_transformers_cache(args.cache_dir)
    from huggingface_hub import notebook_login
    notebook_login(token=args.hf_token)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels, device_map="auto",
    torch_dtype=torch.bfloat16)

    # Function to preprocess the data
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt").to("cuda")

    # Load data
    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)
    test_df = pd.read_csv(args.test_data)

    # Shuffle and merge training and validation data for training
    train_df = shuffle(train_df)
    val_df = shuffle(val_df)
    test_df = shuffle(test_df)
    train_df = pd.concat([train_df, val_df], axis=0)

    # Convert DataFrames into Hugging Face datasets
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    # Tokenize data
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)

    # Remove original text to reduce dataset size
    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_test = tokenized_test.remove_columns(["text"])

    # Data collator to dynamically pad the batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    # Trainer object initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Predict on the test set
    predictions = trainer.predict(tokenized_test)
    pred_probs = softmax(predictions.predictions, axis=1)
    pred_labels = np.argmax(pred_probs, axis=1)

    # Save predictions and model
    predictions_df = pd.DataFrame(pred_probs, columns=['class_0_prob', 'class_1_prob', 'class_2_prob'])
    predictions_df['predicted_label'] = pred_labels
    predictions_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

    model.save_pretrained(os.path.join(args.output_dir, 'model'))

    # Load and compute metrics
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    accuracy = accuracy_metric.compute(predictions=pred_labels, references=predictions.label_ids)
    f1_score = f1_metric.compute(predictions=pred_labels, references=predictions.label_ids, average="weighted")

    print(f'Accuracy: {accuracy["accuracy"]}')
    print(f'F1 Score: {f1_score["f1"]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to classify depression from text data.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data CSV file.")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b-it", help="Model identifier for tokenizer and model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save training outputs.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token for authentication.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU device ID.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenizer.")
    parser.add_argument("--num_labels", type=int, default=3, help="Number of labels for classification task.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for caching models and tokenizer.")

    args = parser.parse_args()
    main(args)
