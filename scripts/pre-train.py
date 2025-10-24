from datasets import load_dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
import torch
import librosa
import evaluate
from dataclasses import dataclass
from typing import Dict, List, Union

# ==============================
# 1️⃣ Load dataset từ 3 file CSV
# ==============================
dataset = load_dataset(
    "csv",
    data_files={
        "train": "features/train.csv",
        "validation": "features/val.csv",
        "test": "features/test.csv"
    }
)

# ==============================
# 2️⃣ Load pre-trained model & processor tiếng Việt
# ==============================
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

# ==============================
# 3️⃣ Chuẩn hóa dữ liệu âm thanh
# ==============================
def prepare(batch):
    audio_array, _ = librosa.load(batch["path"], sr=16000)
    batch["input_values"] = processor(audio_array, sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

dataset = dataset.map(prepare)

# ==============================
# 4️⃣ Data collator đặc biệt cho CTC
# ==============================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        # Replace padding with -100 để CTC bỏ qua khi tính loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# ==============================
# 5️⃣ Cấu hình huấn luyện
# ==============================
args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    num_train_epochs=5,
    save_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    fp16=True,  # dùng float16 nếu GPU hỗ trợ (VD: RTX)
    gradient_accumulation_steps=2,
)

# ==============================
# 6️⃣ Metric: WER (Word Error Rate)
# ==============================
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)

    pred_str = processor.batch_decode(pred_ids)
    label_ids = pred.label_ids
    # Bỏ -100 trước khi decode
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ==============================
# 7️⃣ Trainer
# ==============================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ==============================
# 8️⃣ Huấn luyện mô hình
# ==============================
trainer.train()

