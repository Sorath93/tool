# Use a pipeline as a high-level helper
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

