import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

df = pd.read_csv('datasets/my-dataset-train.csv')

texts = df['text'].tolist()
labels = df['label'].tolist()

# Split the dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the texts and create input features
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create PyTorch datasets
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# weighted sampling during training
#---------------------------------------------------------------------------------------------------
class_weights = class_weight.compute_class_weight(class_weight='balanced',                        #-
                                                  classes=np.unique(train_labels),                #-
                                                  y=train_labels)                                 #-
class_weights = torch.tensor(class_weights, dtype=torch.float)                                    #-
                                                                                                  #-
# Create a weighted random sampler                                                                #-
sampler = WeightedRandomSampler(weights=[class_weights[label] for label in train_labels],         #-
                                num_samples=len(train_dataset),                                   #-
                                replacement=True)                                                 #-
                                                                                                  #-
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)                          #-
#---------------------------------------------------------------------------------------------------
# Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Set up the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)

# Fine-tune the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Device: {device}')
model.to(device)

'''
# training
num_epochs = 3
progress_bar = tqdm(range(num_epochs * len(train_loader)))

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.update(1)
        progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        progress_bar.set_postfix(loss=loss.item())

    model.eval()
    # Evaluate the model on the validation set if desired


# Save the fine-tuned model
model_save_path = 'saved_models/fine_tuned_bert_sentiment'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)


test_df = pd.read_csv("datasets/my-dataset-validation.csv")

# Prepare your test dataset
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

# Tokenize the test texts and create input features
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Create a PyTorch dataset for testing
test_dataset = SentimentDataset(test_encodings, test_labels)

# Create a data loader for testing
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Set the model to evaluation mode
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# Perform testing
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        predicted_labels = torch.argmax(logits, dim=1).tolist()
        true_labels.extend(labels.tolist())
        predictions.extend(predicted_labels)

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
'''

model_save_path = 'saved_models/fine_tuned_bert_sentiment'
loaded_model = BertForSequenceClassification.from_pretrained(model_save_path)
loaded_tokenizer = BertTokenizer.from_pretrained(model_save_path)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model.to(device)

# Set the model to evaluation mode
loaded_model.eval()


# Perform inference
text1 = "Apple is going up!"
inputs1 = loaded_tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
inputs1 = {k: v.to(device) for k, v in inputs1.items()}
outputs1 = loaded_model(**inputs1)
logits1 = outputs1.logits
predicted_class1 = torch.argmax(logits1, dim=1).item()

sentiment_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
predicted_sentiment = sentiment_labels[predicted_class1]

print(f"Text: {text1}")
print(logits1[0].tolist())
print(f"Predicted Sentiment: {predicted_sentiment}\n")

# --------------- second text
text2 = "Tesla doesn't look too good today."
inputs2 = loaded_tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
inputs2 = {k: v.to(device) for k, v in inputs2.items()}
outputs2 = loaded_model(**inputs2)
logits2 = outputs2.logits
predicted_class2 = torch.argmax(logits2, dim=1).item()

sentiment_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
predicted_sentiment = sentiment_labels[predicted_class2]

print(f"Text: {text2}")
print(logits2[0].tolist())
print(f"Predicted Sentiment: {predicted_sentiment}\n")

print(logits1 + logits2)


aapl_tweet_df = pd.read_csv('datasets/preprocessed-evaluation-tweets/data_too_big/aapl-full.csv')
aapl_tweet_df = aapl_tweet_df.iloc[:100000]
tsla_tweet_df = pd.read_csv('datasets/preprocessed-evaluation-tweets/tsla.csv')

aapl_price_df = pd.read_csv('stock_prices/AAPL_prices_2020-03-01_to_2020-08-01.csv')
tsla_price_df = pd.read_csv('stock_prices/TSLA_prices_2022-01-01_to_2024-04-11.csv')

# aapl_tweet_df['char_count'] = aapl_tweet_df['message'].apply(len)
# count_freq = aapl_tweet_df['char_count'].value_counts()

# for length, count in count_freq.items():
#     print(f'length: {length}, Count: {count}')
#     if count < 40:
#         break


def is_more(numbers, index, how_much_more):
    num1, num2 = numbers[:index] + numbers[index+1:]
    max_other = max(num1, num2)
    return numbers[index] >= max_other + how_much_more


y_true = []
y_pred = []

y_true_threshold_1 = []
y_pred_threshold_1 = []

y_true_threshold_2 = []
y_pred_threshold_2 = []

y_true_threshold_3 = []
y_pred_threshold_3 = []

y_true_threshold_4 = []
y_pred_threshold_4 = []

y_true_threshold_5 = []
y_pred_threshold_5 = []

for y in range(3, 8):
    month = str(y).zfill(2)
    for i in range(1, 31):
        if y == 7 and i == 23:
            break
        date = str(i).zfill(2)
        date_string = f'2020-{month}-{date}'
        day_exist = False

        # check if prices fall or rise
        for index, Date in enumerate(aapl_price_df['Date']):
            if date_string == Date[:10]:
                if aapl_price_df['Close'][index+1] > aapl_price_df['Close'][index]:
                    y_true.append(1)
                    y_true_threshold_1.append(1)
                    y_true_threshold_2.append(1)
                    y_true_threshold_3.append(1)
                    y_true_threshold_4.append(1)
                    y_true_threshold_5.append(1)
                else:
                    y_true.append(-1)
                    y_true_threshold_1.append(-1)
                    y_true_threshold_2.append(-1)
                    y_true_threshold_3.append(-1)
                    y_true_threshold_4.append(-1)
                    y_true_threshold_5.append(-1)
                day_exist = True
        if day_exist == False:
            continue
        else:
            day_exist = False

        
        # calculate sentiment on the day
        # summing logits with only highest value, divide by number of instances
        tweet_sentiments = torch.zeros(loaded_model.num_labels).to(device)
        tweet_sentiments_number = [0,0,0]

        for index, Date in enumerate(aapl_tweet_df['datetime']):
            formatted_date = Date[:10]
            if date_string == formatted_date:
                text = aapl_tweet_df['message'][index]
                if len(text) > 90:
                    continue
                    text = text[:120]
                inputs = loaded_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = loaded_model(**inputs)
                logits = outputs.logits
                # totaling sentiment confidence
                mask = torch.zeros_like(logits)
                mask[torch.arange(logits.size(0)), logits.argmax(dim=1)] = 1
                tweet_sentiments += (logits * mask).squeeze()
                # totalling number of cases
                pclass = torch.argmax(logits, dim=1).item()
                tweet_sentiments_number[pclass] += 1


        if tweet_sentiments_number == [0,0,0]:
            y_true.pop()
            y_true_threshold_1.pop()
            y_true_threshold_2.pop()
            y_true_threshold_3.pop()
            y_true_threshold_4.pop()
            y_true_threshold_5.pop()
            continue
        
        print(date_string)
        print(f'Sentiments: {tweet_sentiments.tolist()}')
        print(f'Instances: {tweet_sentiments_number}')

        final_list = []
        for i in range(len(tweet_sentiments)):
            final_list.append(tweet_sentiments.tolist()[i] / tweet_sentiments_number[i])

        predicted_class = final_list.index(max(final_list))
        print(f'final: {final_list}')
        print(predicted_class)
        

        sentiment_labels = {0: -1, 1: 1, 2: 0}
        predicted_sentiment = sentiment_labels[predicted_class]

        if predicted_sentiment in [-1,1]:
            y_pred.append(predicted_sentiment)
            # thresholding 0.05, 0.1, 0.15, 0.2, 0.25
            if is_more(final_list, predicted_class, 0.05):
                y_pred_threshold_1.append(predicted_sentiment)
            else:
                y_true_threshold_1.pop()

            if is_more(final_list, predicted_class, 0.1):
                y_pred_threshold_2.append(predicted_sentiment)
            else:
                y_true_threshold_2.pop()

            if is_more(final_list, predicted_class, 0.15):
                y_pred_threshold_3.append(predicted_sentiment)
            else:
                y_true_threshold_3.pop()

            if is_more(final_list, predicted_class, 0.20):
                y_pred_threshold_4.append(predicted_sentiment)
            else:
                y_true_threshold_4.pop()
            
            if is_more(final_list, predicted_class, 0.25):
                y_pred_threshold_5.append(predicted_sentiment)
            else:
                y_true_threshold_5.pop()


        else:
            y_true.pop()
            y_true_threshold_1.pop()
            y_true_threshold_2.pop()
            y_true_threshold_3.pop()
            y_true_threshold_4.pop()
            y_true_threshold_5.pop()
        


print('Apple')
print(f'y_true: {y_true}')
print(f'y_pred: {y_pred}')

aapl_f1 = f1_score(y_true, y_pred, average='macro')
aapl_accuracy = accuracy_score(y_true, y_pred)

print(f'f1: {aapl_f1}')
print(f'accuracy: {aapl_accuracy}\n')


# 0.05
print(f'y_true_thresh1: {y_true_threshold_1}')
print(f'y_pred_thresh1: {y_pred_threshold_1}')

aapl_f1_1 = f1_score(y_true_threshold_1, y_pred_threshold_1, average='macro')
aapl_accuracy_1 = accuracy_score(y_true_threshold_1, y_pred_threshold_1)

print(f'f1: {aapl_f1_1}')
print(f'accuracy: {aapl_accuracy_1}\n')



# 0.1
print(f'y_true_thresh2: {y_true_threshold_2}')
print(f'y_pred_thresh2: {y_pred_threshold_2}')

aapl_f1_2 = f1_score(y_true_threshold_2, y_pred_threshold_2, average='macro')
aapl_accuracy_2 = accuracy_score(y_true_threshold_2, y_pred_threshold_2)

print(f'f1: {aapl_f1_2}')
print(f'accuracy: {aapl_accuracy_2}\n')


# 0.15
print(f'y_true_thresh3: {y_true_threshold_3}')
print(f'y_pred_thresh3: {y_pred_threshold_3}')

aapl_f1_3 = f1_score(y_true_threshold_3, y_pred_threshold_3, average='macro')
aapl_accuracy_3 = accuracy_score(y_true_threshold_3, y_pred_threshold_3)

print(f'f1: {aapl_f1_3}')
print(f'accuracy: {aapl_accuracy_3}\n')


# 0.2
print(f'y_true_thresh4: {y_true_threshold_4}')
print(f'y_pred_thresh4: {y_pred_threshold_4}')

aapl_f1_4 = f1_score(y_true_threshold_4, y_pred_threshold_4, average='macro')
aapl_accuracy_4 = accuracy_score(y_true_threshold_4, y_pred_threshold_4)

print(f'f1: {aapl_f1_4}')
print(f'accuracy: {aapl_accuracy_4}\n')


# 0.25
print(f'y_true_thresh2: {y_true_threshold_5}')
print(f'y_pred_thresh2: {y_pred_threshold_5}')

aapl_f1_5 = f1_score(y_true_threshold_5, y_pred_threshold_5, average='macro')
aapl_accuracy_5 = accuracy_score(y_true_threshold_5, y_pred_threshold_5)

print(f'f1: {aapl_f1_5}')
print(f'accuracy: {aapl_accuracy_5}\n')