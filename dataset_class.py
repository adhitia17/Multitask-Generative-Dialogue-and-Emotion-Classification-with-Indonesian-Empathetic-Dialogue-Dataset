import pandas as pd
import os

emotion_relations = {
    "excited":      ["excited", "surprised", "joyful"],
    "afraid":       ["afraid", "terrified", "anxious", "apprehensive"],
    "disgusted":    ["disgusted", "embarrassed", "guilty", "ashamed"],
    "annoyed":      ["angry", "annoyed", "jealous", "furious"],
    "grateful":     ["faithful", "trusting", "grateful", "caring", "hopeful"],
    "disappointed": ["sad", "disappointed", "devastated", "lonely", "nostalgic", "sentimental"],
    "impressed":    ["proud", "impressed", "content"],
    "prepared":     ["anticipating", "prepared", "confident"]
}
granular_to_main_emotion = {}
for main_emotion, granular_list in emotion_relations.items():
    for granular in granular_list:
        granular_to_main_emotion[granular.lower().strip()] = main_emotion.lower().strip()

def preprocess_dataframe(df_path: str, output_dir: str, file_prefix: str, is_train: bool=False):
    print(f"\nProcessing {df_path}...")
    try:
        df = pd.read_csv(df_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading {df_path}: {e}")
        return pd.DataFrame()

    if is_train:
        drops = [c for c in ['Unnamed: 5', 'Unnamed: 6'] if c in df.columns]
        if drops:
            df = df.drop(columns=drops)

    required_cols = ['conv_id', 'utterance_idx', 'prompt', 'utterance', 'context']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV {df_path} must contain column '{col}'")

    df['prompt']    = df['prompt'].fillna('').astype(str)
    df['utterance'] = df['utterance'].fillna('').astype(str)
    df['context']   = df['context'].fillna('').astype(str).str.strip()
    df['utterance_idx'] = pd.to_numeric(df['utterance_idx'], errors='coerce').fillna(0).astype(int)

    processed_samples = []
    cnt_dialogue = 0
    cnt_emotion = 0

    seen_prompts = set()
    for idx, row in df.iterrows():
        prompt_text = row['prompt'].strip()
        if not prompt_text:
            continue
        if prompt_text in seen_prompts:
            continue
        seen_prompts.add(prompt_text)

        raw_emotion_label = row['context'].lower().strip()
        if raw_emotion_label:
            emotion_input  = f"emosi: {prompt_text}"
            emotion_target = raw_emotion_label
            processed_samples.append({
                'input_text': emotion_input,
                'target_text': emotion_target,
                'task': 'Emotion'
            })
            cnt_emotion += 1

    for conv_id, group in df.groupby("conv_id"):
        group = group.sort_values(by="utterance_idx").reset_index(drop=True)
        n_turns = len(group)
        for i in range(n_turns):
            if i % 2 == 1:
                history_utts = group.loc[:i-1, 'utterance'].tolist()
                dialog_history = ""
                for j, utt in enumerate(history_utts):
                    speaker = "User" if (j % 2 == 0) else "Bot"
                    dialog_history += f"{speaker}: {utt} [SEP] "

                bot_response = group.loc[i, 'utterance']
                dialogue_input  = f"dialog: {dialog_history}Bot:"
                dialogue_target = bot_response

                processed_samples.append({
                    'input_text': dialogue_input,
                    'target_text': dialogue_target,
                    'task': 'Dialogue'
                })
                cnt_dialogue += 1

    print("Finished preprocessing:")
    print(f"  Generated Emotion samples: {cnt_emotion}")
    print(f"  Generated Dialogue samples: {cnt_dialogue}")

    processed_df = pd.DataFrame(processed_samples)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{file_prefix}_datasetclass.csv")
    if not processed_df.empty:
        processed_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved processed data to: {output_file}")
    else:
        print(f"No samples created; {output_file} was not created.")

    return processed_df

if __name__ == "__main__":
    base_input_dir = "dataset/translated"
    output_dir = "dataset/datasetclass"

    preprocess_dataframe(
        os.path.join(base_input_dir, "train_translated.csv"),
        output_dir,
        "train",
        is_train=True
    )
    preprocess_dataframe(
        os.path.join(base_input_dir, "valid_translated.csv"),
        output_dir,
        "valid",
        is_train=False
    )
    preprocess_dataframe(
        os.path.join(base_input_dir, "test_translated.csv"),
        output_dir,
        "test",
        is_train=False
    )
