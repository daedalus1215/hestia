from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments


# Load and split CSV into train/test/validation.
# Create a Hugging Face dataset.
# Define mappings for sentiment labels.
# Load BERT with proper configuration and move it to GPU.

df = pd.read_csv('twitter_multi_class_sentiment.csv')
train, test = train_test_split(df, test_size=0.3, stratify=df['label'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label'])

dataset = DatasetDict({
    'train': Dataset.from_pandas(train, preserve_index=False),
    'test': Dataset.from_pandas(test, preserve_index=False),
    'validation': Dataset.from_pandas(validation, preserve_index=False)
})

model_ckpt = "bert-base-uncased"

label2id = { x['label_name']: x['label'] for x in dataset['train']}
id2label = {v: k for k, v in label2id.items()}

num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)

batch_size = 64
training_dir = "bert_base_train_dir"

train_args = TrainingArguments(
    output_dir=training_dir, 
    overwrite_output_dir=True,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    disable_tqdm=False,
)


#### Compute Metrics
#### Compute Metrics - Evaluate
#### Compute Metrics - Sklearn

import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics_evaluate(eval_pred):
    f""" Compute metrics using evaluate library. 
    eval_pred is a tuple of predictions and labels, pass automatically by the trainer. 
    Predictions are the raw logits (model outputs before softmax). 
    Labels are the ground-truth numeric class IDs.
    np.argmax(predictions, axis=1) converts the logits to predicted class IDs by picking the index of the max logit (the most likely class).
    accuracy.compute() calls the built-in hugging face metric to compute accuracy. returns a dict like {"accuracy": 0.9}
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1
    }

#### Convert raw logits → class predictions (argmax)
#### Compare predictions vs. ground-truth labels
#### Return a dict of metric results to the Trainer
#### The difference is mainly in what library computes the metrics:
#### evaluate → simpler, built-in metric modules
#### sklearn → full control, multiple metrics (accuracy + F1 here)



## Build Model and Trainer

#### Create emotion_encoded dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    """
    This tokenizer knows:
    How to split text into tokens (words/subwords)
    How to map those tokens into integer IDs from the model’s vocabulary
    Which special tokens to add (like [CLS], [SEP], [PAD], etc.)
    The maximum sequence length for that model.

    This function takes a batch (a dictionary from the dataset).
    batch['text'] is the column with the tweet text.
    It calls the tokenizer on that text.
    """
    temp = tokenizer(batch['text'], padding=True, truncation=True)
    return temp

emotion_encoded = dataset.map(tokenize, batched=True, batch_size=None)


#### Build Trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=train_args,
    compute_metrics=compute_metrics,
    train_dataset=emotion_encoded['train'],
    eval_dataset = emotion_encoded['validation'],
    tokenizer=tokenizer
)

trainer.train()


#### Model Evaluation
"""
   Take the test dataset (emotion_encoded['test'])
   Tokenize and batch it automatically
   Feed it through the trained model
   Collect:
    - the raw logits (model outputs before softmax)
    - the true labels
    - and compute any metrics we defined in compute_metrics
    It returns a PredictionOutput object that contains three things:

    PredictionOutput(
        predictions = array([...]),   # logits from model, shape (num_samples, num_classes)
        label_ids   = array([...]),   # true labels
        metrics     = {...}           # accuracy, F1, loss, etc.
    )
"""
preds_output = trainer.predict(emotion_encoded['test'])
"""
{
  "test_loss": 0.46,
  "test_accuracy": 0.87,
  "test_f1": 0.85,
  "test_runtime": 12.34,
  "test_samples_per_second": 200.5
}
"""
preds_output.metrics


##### Extract predicted and true labels manually
y_pred = np.argmax(preds_output.predictions, axis=1) # Predicted labels - picks the class with the highest score for each sample
y_true = emotion_encoded['test'][:]['label'] # Model's predicted class IDs. Extracts all ground-truth labels from the test dataset


"""
- Precision — how many predicted positives were correct
(out of all examples the model predicted as that class)
- Recall — how many actual positives were found
(out of all examples that truly belonged to that class).
- F1-score — harmonic mean of precision and recall
- Support — number of samples per class
- Accuracy — overall proportion of correct predictions
"""
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
print(label2id)


##### plot confusion matrix

# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt

# cm = confusion_matrix(y_true, y_pred)

# plt.figure(figsize=(5,5))
# sns.heatmap(cm, annot=True, xticklabels=label2id.keys(), yticklabels=label2id.keys())
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()



### Build Prediction Function and Store Model
def get_predictions(text):
    """
    inference prediction function. It takes any text string, runs it through the fine-tuned model, and returns the predicted label name (e.g. "joy", "sadness", etc.).
    """

    """
    This tokenizer is the same one we trained with to convert the input text into model-ready tensors. It should return something like:

    {
        'input_ids': tensor([[101, 2026, 3899, 2003, 2307, 102]]),
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])
    }
    """
    input_encoded = tokenizer(text, return_tensors='pt').to(device)


    """
    - Disables gradient tracking — this is an inference-only context.
    - It saves memory and speeds up computation since you’re not training or  back-propagating.
    - Used only for prediction / evaluation, not training.
    """
    with torch.no_grad():
        outputs = model(**input_encoded)

    logits = outputs.logits

    pred = torch.argmax(logits, dim=1).item()
    return id2label[pred]

get_predictions("I was so happy")

trainer.save_model('bert-base-uncased-serntiment-model')



### Use Pipeline for prediction
from transformers import Pipeline
classifier = pipeline('text-classification', model='bert-base-uncased-sentiment-model')
classifier([text, 'hello, how are you?", "love you'])