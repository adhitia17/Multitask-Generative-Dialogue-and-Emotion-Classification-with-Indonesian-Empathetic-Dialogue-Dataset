import os

# GPU Setup (can be uncommented if specific GPU visibility is needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["NCCL_P2P_DISABLE"] = "1"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
from sklearn.metrics import accuracy_score, f1_score

# Import Hugging Face evaluate library
import evaluate

# --- Define constants and configurations ---
MODEL_NAME = "muchad/idt5-base" # Used for tokenizer, model config loaded from checkpoint
DEFAULT_PROCESSED_DATA_DIR = "dataset/processed_data_csv"
DEFAULT_CHECKPOINT_DIR = "./t5_multitask_checkpoints"

# --- GPU Device Setup (after torch import) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✅ PyTorch CUDA device selected for evaluation: {device}")
    print(f"✅ Current CUDA device for evaluation: {torch.cuda.current_device()}")
else:
    print("✅ Using CPU for evaluation:", device)


# --- Helper Functions (Dataset, Evaluation) ---
class MultitaskDatasetCSV(Dataset):
    def __init__(self, tokenizer, file_path, max_seq_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_target_length = max_target_length
        self.samples = []
        print(f"Loading data from {file_path}...")
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
            required_cols = ["input_text", "target_text", "task"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV file {file_path} must contain columns: {required_cols}")
            
            df["input_text"] = df["input_text"] .fillna("").astype(str)
            df["target_text"] = df["target_text"] .fillna("").astype(str)
            df["task"] = df["task"] .fillna("unknown").astype(str)

            for index, row in df.iterrows():
                self.samples.append({
                    "input_text": row["input_text"],
                    "target_text": row["target_text"],
                    "task": row["task"]
                })
            print(f"Loaded {len(self.samples)} samples from {file_path}.")
        except FileNotFoundError:
            print(f"Error: File not found {file_path}")
            raise
        except Exception as e:
            print(f"Error reading or processing CSV file {file_path}: {e}")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["input_text"]
        target_text = sample["target_text"]

        input_encoding = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        # For evaluation, we only need the original target text for metrics, not tokenized labels for loss calculation by default
        # However, if loss calculation during eval is desired, target_encoding is needed.
        # labels = target_encoding["input_ids"].squeeze(0)
        # labels[labels == self.tokenizer.pad_token_id] = -100

        input_ids = input_encoding["input_ids"].squeeze(0)
        attention_mask = input_encoding["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "labels": labels, # Only if calculating loss during eval
            "task": sample.get("task", "unknown"),
            "original_input": input_text,
            "original_target": target_text
        }

def evaluate_model(model, dataloader, current_device, tokenizer, max_target_length_eval):
    model.eval()
    all_preds_emotion, all_labels_emotion = [], []
    all_preds_dialogue, all_labels_dialogue = [], []
    
    # Load metrics from Hugging Face evaluate
    try:
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
    except Exception as e:
        print(f"Error loading metrics from Hugging Face evaluate: {e}")
        print("Please ensure you have an internet connection and the necessary libraries (evaluate, sacrebleu, rouge_score) installed.")
        return None # Or handle error appropriately

    print("Starting model evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(current_device)
            attention_mask = batch["attention_mask"].to(current_device)
            # labels = batch["labels"].to(current_device) # If calculating loss
            tasks = batch["task"]
            original_targets = batch["original_target"]

            # Generate predictions for all tasks
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length_eval,
                num_beams=4,
                early_stopping=True
            )
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i in range(len(tasks)):
                pred_text = decoded_preds[i].strip()
                true_label = original_targets[i].strip()
                
                if tasks[i] == "Emotion":
                    all_preds_emotion.append(pred_text.lower())
                    all_labels_emotion.append(true_label.lower())
                elif tasks[i] == "Dialogue":
                    all_preds_dialogue.append(pred_text)
                    all_labels_dialogue.append(true_label)
            
            if batch_idx > 0 and batch_idx % 100 == 0:
                 print(f"  Evaluated batch {batch_idx}/{len(dataloader)}")

    # --- Emotion Metrics ---
    if all_labels_emotion:
        if len(all_preds_emotion) == len(all_labels_emotion):
            emotion_accuracy = accuracy_score(all_labels_emotion, all_preds_emotion)
            emotion_f1 = f1_score(all_labels_emotion, all_preds_emotion, average="weighted", zero_division=0)
            print(f"\n--- Emotion Classification Metrics ---")
            print(f"  Accuracy: {emotion_accuracy:.4f}")
            print(f"  F1 Score (Weighted): {emotion_f1:.4f}")
        else:
            print("Warning: Mismatch in length of emotion predictions and labels. Skipping emotion metrics.")
    else:
        print("\nNo emotion classification samples found in this dataset split.")

    # --- Dialogue Metrics (BLEU & ROUGE) ---
    if all_labels_dialogue:
        print(f"\n--- Dialogue Metrics ---")
        # BLEU expects references as a list of lists of strings
        bleu_references = [[ref] for ref in all_labels_dialogue]
        try:
            bleu_results = bleu_metric.compute(predictions=all_preds_dialogue, references=bleu_references)
            print(f"  BLEU Score: {bleu_results['bleu']:.4f}")
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")

        try:
            rouge_results = rouge_metric.compute(predictions=all_preds_dialogue, references=all_labels_dialogue)
            print(f"  ROUGE-1: {rouge_results['rouge1']:.4f}")
            print(f"  ROUGE-2: {rouge_results['rouge2']:.4f}")
            print(f"  ROUGE-L: {rouge_results['rougeL']:.4f}")
            print(f"  ROUGE-Lsum: {rouge_results['rougeLsum']:.4f}")
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
        
        print("\nSample dialogue Predictions (first 5):")
        for i in range(min(5, len(all_preds_dialogue))):
            print(f"  Reference: {all_labels_dialogue[i]}")
            print(f"  Predicted: {all_preds_dialogue[i]}\n")
    else:
        print("\nNo Dialogue samples found in this dataset split.")

    return  # Atau return dict metrics, jika diperlukan

# --- Main Evaluation Function ---
def main_eval(args):
    current_device = device

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

    eval_file_path = os.path.join(args.processed_data_dir, args.eval_file_name)

    print("Creating evaluation dataset and dataloader...")
    eval_dataset = MultitaskDatasetCSV(tokenizer, eval_file_path, args.max_seq_length, args.max_target_length)

    if len(eval_dataset) == 0:
        print(f"Evaluation dataset is empty. Please check {eval_file_path}. Exiting.")
        return

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Eval Dataloader: {len(eval_dataloader)} batches of size {args.batch_size}")

    if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path \"{args.checkpoint_path}\" not provided or does not exist. Exiting.")
        return

    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    checkpoint = torch.load(args.checkpoint_path, map_location=current_device)
    
    model_state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in model_state_dict.keys()):
        print("Removing \"module.\" prefix from state_dict keys.")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    
    model.to(current_device)
    print(f"Model moved to device: {next(model.parameters()).device}")

    evaluate_model(model, eval_dataloader, current_device, tokenizer, args.max_target_length)

    print("\nEvaluation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained T5 multitask model using Hugging Face evaluate for BLEU/ROUGE.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.pt file) to evaluate.")
    parser.add_argument("--processed_data_dir", type=str, default=DEFAULT_PROCESSED_DATA_DIR, help="Directory containing the processed CSV evaluation file.")
    parser.add_argument("--eval_file_name", type=str, default="test_processed.csv", help="Name of the evaluation CSV file (e.g., valid_processed.csv or test_processed.csv).")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length for input.")
    parser.add_argument("--max_target_length", type=int, default=128, help="Maximum sequence length for target output during generation.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation. Adjust based on GPU memory.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes for DataLoader.")

    args = parser.parse_args()
    main_eval(args)
