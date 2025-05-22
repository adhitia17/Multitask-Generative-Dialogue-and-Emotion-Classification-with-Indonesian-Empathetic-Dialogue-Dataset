import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
import glob
from sklearn.metrics import accuracy_score, f1_score
import argparse
import json
import shutil

MODEL_NAME = "muchad/idt5-base"
DEFAULT_PROCESSED_DATA_DIR = "dataset/datasetclass_nogroup"
DEFAULT_CHECKPOINT_DIR = "./t5_multitask_checkpoints"
DEFAULT_HF_MODEL_DIR = "./t5_multitask_hf_model"
DEFAULT_TRAINER_LOG_FILE = "trainer_log.json"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"PyTorch CUDA device selected: {device} (GPU ID likely 7 due to CUDA_VISIBLE_DEVICES)")
    print(f"Current CUDA device according to PyTorch: {torch.cuda.current_device()}")
else:
    print(f"Using device: {device}")

class MultitaskDatasetCSV(Dataset):
    def __init__(self, tokenizer, file_path, max_seq_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_target_length = max_target_length
        self.samples = []
        print(f"Loading and tokenizing data from {file_path}...")
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            required_cols = ['input_text', 'target_text', 'task']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV file {file_path} must contain columns: {required_cols}")

            df['input_text'] = df['input_text'].fillna('').astype(str)
            df['target_text'] = df['target_text'].fillna('').astype(str)
            df['task'] = df['task'].fillna('unknown').astype(str)

            task_counts = df['task'].value_counts()
            print("Task distribution in dataset:")
            for task, count in task_counts.items():
                print(f"  {task}: {count} samples")

            for _, row in df.iterrows():
                self.samples.append({
                    'input_text': row['input_text'],
                    'target_text': row['target_text'],
                    'task': row['task']
                })
            print(f"Loaded {len(self.samples)} samples from {file_path}.")
        except FileNotFoundError:
            print(f"Error: File not found {file_path}")
        except Exception as e:
            print(f"Error reading or processing CSV file {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample['input_text']
        target_text = sample['target_text']

        input_encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        )

        input_ids = input_encoding['input_ids'].squeeze(0)
        attention_mask = input_encoding['attention_mask'].squeeze(0)
        labels = target_encoding['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'task': sample.get('task', 'unknown'),
            'original_target': target_text
        }

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, current_device):
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch-*.pt"))
        if checkpoints:
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            latest_checkpoint = checkpoints[0]

    start_epoch = 0
    best_val_metric = float('inf')
    saved_checkpoints_info = []
    training_logs = []

    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=current_device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint.get('best_val_metric', float('inf'))
        saved_checkpoints_info = checkpoint.get('saved_checkpoints_info', [])
        log_file_path = os.path.join(checkpoint_dir, DEFAULT_TRAINER_LOG_FILE)
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r') as f:
                    training_logs = json.load(f)
                print(f"Loaded existing training logs from {log_file_path}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {log_file_path}. Starting with empty logs.")
        print(f"Resumed from epoch {start_epoch-1}, best validation metric so far: {best_val_metric:.4f}")
    else:
        print("No checkpoint found. Starting training from scratch.")
    return start_epoch, best_val_metric, saved_checkpoints_info, training_logs

def save_checkpoint(epoch, model, optimizer, scheduler, current_val_metric, best_val_metric, checkpoint_dir, saved_checkpoints_info, save_top_k):
    is_best = current_val_metric < best_val_metric
    if is_best:
        best_val_metric = current_val_metric

    current_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}-metric-{current_val_metric:.4f}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metric': current_val_metric,
        'best_val_metric': best_val_metric,
        'saved_checkpoints_info': saved_checkpoints_info
    }, current_checkpoint_path)
    print(f"Saved checkpoint: {current_checkpoint_path}")

    saved_checkpoints_info.append((current_val_metric, current_checkpoint_path))
    saved_checkpoints_info.sort(key=lambda x: x[0])

    if len(saved_checkpoints_info) > save_top_k:
        _, worst_checkpoint_path = saved_checkpoints_info.pop()
        if os.path.exists(worst_checkpoint_path):
            try:
                os.remove(worst_checkpoint_path)
                print(f"Removed worse checkpoint: {worst_checkpoint_path}")
            except OSError as e:
                print(f"Error removing checkpoint {worst_checkpoint_path}: {e}")

    return best_val_metric, saved_checkpoints_info, is_best

def save_model_hf_format(model, tokenizer, hf_model_dir, epoch, current_val_metric):
    best_dir = os.path.join(hf_model_dir, "best")
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    os.makedirs(best_dir)

    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    config_info = {
        "base_model": MODEL_NAME,
        "epoch": epoch,
        "val_metric": float(f"{current_val_metric:.4f}"),
        "save_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(best_dir, "training_info.json"), 'w') as f:
        json.dump(config_info, f, indent=4)

    print(f"Saved best model and tokenizer in Hugging Face format to {best_dir}")

def evaluate_epoch(model, dataloader, current_device, tokenizer, max_target_length_eval):
    model.eval()
    total_eval_loss = 0

    all_preds_emotion = []
    all_labels_emotion = []
    all_preds_dialogue = []
    all_labels_dialogue = []

    print("Starting validation epoch...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(current_device)
            attention_mask = batch['attention_mask'].to(current_device)
            labels = batch['labels'].to(current_device)
            tasks = batch['task']
            original_targets = batch['original_target']

            if batch_idx == 0:
                print(f"DEBUG (eval): input_ids device: {input_ids.device}")
                print(f"DEBUG (eval): attention_mask device: {attention_mask.device}")
                print(f"DEBUG (eval): labels device: {labels.device}")
                print(f"DEBUG (eval): model device (example param): {next(model.parameters()).device}")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()

            for i in range(len(tasks)):
                pred_ids = model.generate(
                    input_ids[i].unsqueeze(0),
                    attention_mask=attention_mask[i].unsqueeze(0),
                    max_length=max_target_length_eval,
                    num_beams=4,
                    early_stopping=True
                )
                pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip().lower()
                true_label = original_targets[i].strip().lower()

                if tasks[i] == 'Emotion':
                    all_preds_emotion.append(pred_text)
                    all_labels_emotion.append(true_label)
                elif tasks[i] == 'Dialogue':
                    all_preds_dialogue.append(pred_text)
                    all_labels_dialogue.append(true_label)

            if batch_idx > 0 and batch_idx % 200 == 0:
                print(f"  Validated batch {batch_idx}/{len(dataloader)}")

    avg_eval_loss = total_eval_loss / len(dataloader)
    print(f"Average Validation Loss: {avg_eval_loss:.4f}")

    emotion_accuracy = 0
    emotion_f1 = 0
    if all_labels_emotion:
        if len(all_preds_emotion) == len(all_labels_emotion):
            emotion_accuracy = accuracy_score(all_labels_emotion, all_preds_emotion)
            emotion_f1 = f1_score(all_labels_emotion, all_preds_emotion, average='weighted', zero_division=0)
            print(f"Emotion Classification Accuracy: {emotion_accuracy:.4f}")
            print(f"Emotion Classification F1 Score (Weighted): {emotion_f1:.4f}")
        else:
            print("Warning: Mismatch in length of emotion predictions and labels. Skipping emotion metrics.")

    return avg_eval_loss, emotion_accuracy, emotion_f1

def train(args):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.hf_model_dir):
        os.makedirs(args.hf_model_dir)

    log_file_path = os.path.join(args.checkpoint_dir, DEFAULT_TRAINER_LOG_FILE)
    current_device = device

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

    train_file_path = os.path.join(args.processed_data_dir, "train_datasetclass.csv")
    valid_file_path = os.path.join(args.processed_data_dir, "valid_datasetclass.csv")

    print("Creating datasets and dataloaders...")
    train_dataset = MultitaskDatasetCSV(tokenizer, train_file_path, args.max_seq_length, args.max_target_length)
    valid_dataset = MultitaskDatasetCSV(tokenizer, valid_file_path, args.max_seq_length, args.max_target_length)

    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        print("Training or validation dataset is empty. Exiting.")
        return

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train Dataloader: {len(train_dataloader)} batches of size {args.batch_size}")
    print(f"Valid Dataloader: {len(valid_dataloader)} batches of size {args.batch_size}")

    print(f"Loading model: {MODEL_NAME}")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(current_device)
    print(f"Model moved to device: {next(model.parameters()).device}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = (len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    start_epoch, best_val_metric, saved_checkpoints_info, training_logs = load_checkpoint(model, optimizer, scheduler, args.checkpoint_dir, current_device)

    print("Starting training loop...")
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(current_device)
            attention_mask = batch['attention_mask'].to(current_device)
            labels = batch['labels'].to(current_device)

            if batch_idx == 0 and epoch == start_epoch:
                print(f"DEBUG (train): input_ids device: {input_ids.device}")
                print(f"DEBUG (train): attention_mask device: {attention_mask.device}")
                print(f"DEBUG (train): labels device: {labels.device}")
                print(f"DEBUG (train): model device (example param): {next(model.parameters()).device}")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            total_train_loss += loss.item() * args.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if batch_idx > 0 and (batch_idx * args.batch_size) % (args.logging_steps * args.batch_size * args.gradient_accumulation_steps) == 0:
                print(f"  Epoch {epoch + 1}/{args.num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, "
                      f"Avg Train Loss (last {args.logging_steps} effective batches): "
                      f"{total_train_loss / (args.logging_steps * args.gradient_accumulation_steps):.4f}")

        avg_train_loss_epoch = total_train_loss / (len(train_dataloader) * args.gradient_accumulation_steps)
        print(f"Epoch {epoch + 1}/{args.num_epochs} - Average Training Loss: {avg_train_loss_epoch:.4f}")

        val_loss, val_emotion_acc, val_emotion_f1 = evaluate_epoch(
            model, valid_dataloader, current_device, tokenizer, args.max_target_length
        )
        print(f"Epoch {epoch + 1} Validation: Loss={val_loss:.4f}, Emotion Acc={val_emotion_acc:.4f}, Emotion F1={val_emotion_f1:.4f}")

        epoch_log = {
            'epoch': epoch + 1,
            'avg_training_loss': round(avg_train_loss_epoch, 4),
            'validation_loss': round(val_loss, 4),
            'validation_emotion_accuracy': round(val_emotion_acc, 4),
            'validation_emotion_f1_score': round(val_emotion_f1, 4)
        }
        training_logs.append(epoch_log)
        with open(log_file_path, 'w') as f:
            json.dump(training_logs, f, indent=4)
        print(f"Saved training log to {log_file_path}")

        best_val_metric, saved_checkpoints_info, is_best = save_checkpoint(
            epoch, model, optimizer, scheduler, val_loss, best_val_metric,
            args.checkpoint_dir, saved_checkpoints_info, args.save_top_k
        )

        if is_best:
            save_model_hf_format(model, tokenizer, args.hf_model_dir, epoch, val_loss)

    print("Training finished.")
    print(f"Checkpoints are saved in: {args.checkpoint_dir}")
    print(f"Best Hugging Face model is saved in: {os.path.join(args.hf_model_dir, 'best')}")
    print(f"Training logs are saved in: {log_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 model for multitask Dialogue and Emotion Classification.")
    parser.add_argument("--processed_data_dir", type=str, default=DEFAULT_PROCESSED_DATA_DIR,
                        help="Directory containing processed CSV files.")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory to save model checkpoints and logs.")
    parser.add_argument("--hf_model_dir", type=str, default=DEFAULT_HF_MODEL_DIR,
                        help="Directory to save models in Hugging Face format.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum input sequence length (tokens).")
    parser.add_argument("--max_target_length", type=int, default=128,
                        help="Maximum target sequence length (tokens).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=40,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating.")
    parser.add_argument("--save_top_k", type=int, default=3,
                        help="Maximum number of checkpoints to keep.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for DataLoader.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Number of batches between logging training loss.")

    args = parser.parse_args()
    train(args)
