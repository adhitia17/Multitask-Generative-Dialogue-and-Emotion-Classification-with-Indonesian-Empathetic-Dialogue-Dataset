import pandas as pd
from tqdm import tqdm
import os
import tarfile
import urllib.request
import shutil
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

base_dir = 'dataset'
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, 'translated'), exist_ok=True)
dataset_dir = os.path.join(base_dir, 'empatheticdialogues')
os.makedirs(dataset_dir, exist_ok=True)

data_files = [
    os.path.join(dataset_dir, 'train.csv'),
    os.path.join(dataset_dir, 'valid.csv'),
    os.path.join(dataset_dir, 'test.csv')
]

if not all(os.path.exists(f) for f in data_files):
    print("Downloading dataset...")
    tar_path = os.path.join(dataset_dir, 'empatheticdialogues.tar.gz')
    url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    urllib.request.urlretrieve(url, tar_path)
    print(f"Dataset downloaded to {tar_path}")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=dataset_dir)
    print("Dataset extracted")
    extracted_dir = os.path.join(dataset_dir, 'empatheticdialogues')
    for filename in ['train.csv', 'valid.csv', 'test.csv']:
        src = os.path.join(extracted_dir, filename)
        dst = os.path.join(dataset_dir, filename)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {filename} to {dst}")
        else:
            print(f"File {filename} not found in {extracted_dir}")
    try:
        os.rmdir(extracted_dir)
    except OSError:
        pass
    print("Data preparation complete")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"

model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="eng_Latn",
    tgt_lang="ind_Latn",
    device=0
)

keep_columns = ['conv_id', 'utterance_idx', 'context', 'prompt', 'utterance']

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-z0-9\s.,?!']", "", text)
    text = text.replace("_comma_", ",")
    text = text.replace("_apostrophe_", "'")
    return text

def process_file(input_path, output_path):
    print(f"Processing: {input_path}")
    try:
        df = pd.read_csv(input_path, on_bad_lines='skip')
        df = df[[col for col in keep_columns if col in df.columns]]
        df['prompt'] = df['prompt'].astype(str)
        df['utterance'] = df['utterance'].astype(str)
        tqdm.pandas()
        df['prompt'] = df['prompt'].progress_apply(lambda x: translator(clean_text(x))[0]['translation_text'])
        df['utterance'] = df['utterance'].progress_apply(lambda x: translator(clean_text(x))[0]['translation_text'])
        df.to_csv(output_path, index=False)
        print(f"Finished writing: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

translated_dir = os.path.join(base_dir, 'translated')
process_file(os.path.join(dataset_dir, 'train.csv'), os.path.join(translated_dir, 'train_translated.csv'))
process_file(os.path.join(dataset_dir, 'valid.csv'), os.path.join(translated_dir, 'valid_translated.csv'))
process_file(os.path.join(dataset_dir, 'test.csv'), os.path.join(translated_dir, 'test_translated.csv'))
