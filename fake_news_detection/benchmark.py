
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1
    }

model_dict = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "mobilebert": "google/mobilebert-uncased",
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_excel('../resources/fake_news.xlsx', dtype={'title': str, 'text': str})
df = df.dropna()
# 70% for training, 20% test, 10% validation
train, test = train_test_split(df, test_size=0.3, stratify=df['label'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label'])

train.shape, test.shape, validation.shape, df.shape

from datasets import Dataset, DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_pandas(train, preserve_index=False),
    "test": Dataset.from_pandas(test, preserve_index=False),
    "validation": Dataset.from_pandas(validation, preserve_index=False)
})

from transformers import Trainer
from transformers import TrainingArguments

def train_model(model_name):
    model_ckpt = model_dict[model_name]
    label2id = {"Real": 0, "Fake": 1}
    id2label = {0: "Real", 1: "Fake"}
    config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    def local_tokenizer(batch):
        temp = tokenizer(batch['title'], padding=True, truncation=True)
        return temp

    encoded_dataset = dataset.map(local_tokenizer, batched=True, batch_size=None)
    training_dir = "train_dir"
    batch_size = 64
    num_train_epochs = 2
    learning_rate = 2e-5
    weight_decay = 0.01
    disable_tqdm = False
    training_args = TrainingArguments(
        output_dir=training_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=encoded_dataset['train'],
        eval_dataset = encoded_dataset['validation'],
        processing_class=tokenizer
    )

    trainer.train()
    preds = trainer.predict(encoded_dataset['test'])
    trainer.save_model("fake_news")
    return preds.metrics

model_performance = {}
for model_name in model_dict:
    temp = train_model(model_name)
    model_performance[model_name] = temp

print(model_performance)
