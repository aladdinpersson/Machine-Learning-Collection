import evaluate
from transformers import Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from datasets import load_dataset, DatasetDict, Audio
# set so we only can see first cuda device 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "sv-SE",
    split="train+validation",
    use_auth_token=False,
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "sv-SE",
    split="test",
    use_auth_token=False,
)

# common_voice = common_voice.remove_columns(
#     [
#         "accent",
#         "age",
#         "client_id",
#         "down_votes",
#         "gender",
#         "locale",
#         "path",
#         "segment",
#         "up_votes",
#     ]
# )

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-tiny", language="sv", task="transcribe"
)

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="sv", task="transcribe"
)

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(example):
    # load and resample audio data from 48 to 16kHz
    audio = example["audio"]

    # compute log-Mel input features from input audio array
    example["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    example["labels"] = tokenizer(example["sentence"]).input_ids
    return example


common_voice = common_voice.map(prepare_dataset, num_proc=8)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths
        # and need different padding methods first treat the audio inputs by
        # simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-swedish",  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=False,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=0,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
