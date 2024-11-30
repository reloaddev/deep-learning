# For colab
!pip install datasets

from collections import Counter
import torch
from datasets import load_dataset
import pandas as pd

# Data preparation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("dair-ai/emotion", "split")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
validation_df = ds["validation"].to_pandas()

# Extract class labels into list of integers

train_dist = train_df['label'].value_counts(normalize=True)
test_dist = test_df['label'].value_counts(normalize=True)
validation_dist = validation_df['label'].value_counts(normalize=True)

print(train_dist)
print(test_dist)
print(validation_dist)

# What is the chance accuracy level?

chance_level_train = (train_dist** 2).sum()
chance_level_test = (test_dist ** 2).sum()
chance_level_val = (validation_dist ** 2).sum()

print("Chance Levels")
print("Train: ", chance_level_train)
print("Test: ", chance_level_test)
print("Validation: ", chance_level_val)

# What would be the accuracy of a classifier
# that only predicts the most common class seen in training?

print("Accuracy of classifier only predicting most common class: ", 13521/40000)


train_labels = train_df["label"].tolist()
label_counts = Counter(train_labels)
total_samples = len(train_labels)
num_classes = len(label_counts)
class_weights = [total_samples / label_counts[i] for i in range(num_classes)]
class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights = class_weights.to(device)


splits = [
    {"label": "Train", "df": train_df},
    {"label": "Test", "df": test_df},
    {"label": "Validation", "df": validation_df}
]
for split in splits:
    text_lengths = split["df"]["text"].map(lambda x: len(x))
    text_lengths_range = text_lengths.max() - text_lengths.min()
    print(f"[{split['label']}] Text Length - Range              :", text_lengths_range)
    text_lengths_mean = text_lengths.mean()
    print(f"[{split['label']}] Text Length - Mean               :", text_lengths_mean)
    text_lengths_std = text_lengths.std()
    print(f"[{split['label']}] Text Length - Std                :", text_lengths_std)


# Extract the texts for all splits and split each text into tokens.
import re
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()
def whitespace_tokenizer(text):
    return text.split()

train_df["text"] = train_df["text"].apply(preprocess_text)
test_df["text"] = test_df["text"].apply(preprocess_text)
validation_df["text"] = validation_df["text"].apply(preprocess_text)
train_df["tokens"] = train_df["text"].apply(lambda x: whitespace_tokenizer(x))
test_df["tokens"] = test_df["text"].apply(lambda x: whitespace_tokenizer(x))
validation_df["tokens"] = validation_df["text"].apply(lambda x: whitespace_tokenizer(x))


# Build a vocabulary (map string to integer) based on train split
from collections import Counter
import torch

counter = Counter()
for sample in train_df["tokens"]:
    counter.update(sample)
vocabulary = {
    '<UNK>': 0,
    '<PAD>': 1,
    **{word: idx + 2 for idx, (word, count) in enumerate(counter.most_common(8000))}
}

# Sequences shorter than max_length, will be filled
# up with <PAD> until they match max_length
def pad_sequence(sequence, max_length=25, pad_value=1):
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence + [pad_value] * (max_length - len(sequence))

# Encode and pad all texts with the defined vocabulary
train_sequences = [pad_sequence([vocabulary.get(token, 0) for token in sample]) for sample in train_df["tokens"]]
test_sequences = [pad_sequence([vocabulary.get(token, 0) for token in sample]) for sample in test_df["tokens"]]
validation_sequences = [pad_sequence([vocabulary.get(token, 0) for token in sample]) for sample in validation_df["tokens"]]

unk_count = sum(sequence.count(0) for sequence in train_sequences)
print(f"Percentage of <UNK> tokens (train): {unk_count / sum(len(seq) for seq in train_sequences) * 100:.2f}%")
unk_count_test = sum(sequence.count(0) for sequence in test_sequences)
print(f"Percentage of <UNK> tokens (test): {unk_count / sum(len(seq) for seq in test_sequences) * 100:.2f}%")
unk_count_validation = sum(sequence.count(0) for sequence in validation_sequences)
print(f"Percentage of <UNK> tokens (test): {unk_count / sum(len(seq) for seq in validation_sequences) * 100:.2f}%")
pad_count = sum(sequence.count(1) for sequence in train_sequences)
print(f"Percentage of <PAD> tokens (train): {pad_count / sum(len(seq) for seq in train_sequences) * 100:.2f}%")
pad_count_test = sum(sequence.count(1) for sequence in test_sequences)
print(f"Percentage of <PAD> tokens (test): {pad_count_test / sum(len(seq) for seq in test_sequences) * 100:.2f}%")
pad_count_validation = sum(sequence.count(1) for sequence in validation_sequences)
print(f"Percentage of <PAD> tokens (test): {pad_count_validation / sum(len(seq) for seq in validation_sequences) * 100:.2f}%")


# Load the data
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Assuming you have labels in your DataFrame
train_labels = train_df["label"].tolist()
test_labels = test_df["label"].tolist()
validation_labels = validation_df["label"].tolist()

# Create dataset instances
train_dataset = TextDataset(train_sequences, train_labels)
test_dataset = TextDataset(test_sequences, test_labels)
validation_dataset = TextDataset(validation_sequences, validation_labels)

# Create DataLoader instances
from torch.utils.data import WeightedRandomSampler

# Calculate sample weights
class_sample_counts = Counter(train_labels)
weights = [1.0 / class_sample_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=32)


# Load pretrained embeddings
import numpy as np

def load_glove_embeddings(filepath, vocab, embedding_dim=100):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if word in vocab:
                embedding_matrix[vocab[word]] = vector
    return embedding_matrix

#####################

import torch
import torch.nn as nn
import torch.nn.init as init

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=128, num_layers=1, num_classes=6, dropout=0.3, pretrained_embeddings=None, freeze_embeddings=True):
        super(GRUModel, self).__init__()

        if pretrained_embeddings is not None:
            # Use pretrained embeddings
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings, padding_idx=1)
        else:
            # Initialize embeddings randomly
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
            init.xavier_uniform_(self.embedding.weight)

        # GRU with dropout
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc = nn.Linear(hidden_size * 2, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.to(next(self.embedding.parameters()).device)  # Ensure x is on the same device as the embedding layer
        x = self.embedding(x)
        h0 = torch.zeros(self.rnn.num_layers * 2, x.size(0), self.rnn.hidden_size).to(x.device)
        out, hn = self.rnn(x, h0)

        # Get the last hidden state
        hidden_state_outputs = out[:, -1, :]

        # Fully connected layers
        hidden_state_outputs = self.fc(hidden_state_outputs)
        hidden_state_outputs = self.relu(hidden_state_outputs)
        hidden_state_outputs = self.dropout(hidden_state_outputs)
        result = self.fc2(hidden_state_outputs)
        return result
    
    #####################
    
    # Train the model
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(
    vocab_size=len(vocabulary),
    embedding_dim=100,
    hidden_size=128,
    num_layers=3,
    num_classes=6,
    dropout=0.5,
    # pretrained_embeddings=embedding_tensor,
    freeze_embeddings=False
).to(device)


epochs = 20
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

train_losses = []
validation_losses = []
train_accuracies = []
validation_accuracies = []


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    avg_train_loss = running_loss / len(dataloader)
    train_accuracy = correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Training Error: \n Accuracy: {train_accuracy * 100:.2f}%, Avg loss: {avg_train_loss:.4f}")


def validation_loop(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss, val_correct = 0, 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            _, predicted = torch.max(pred.data, 1)
            val_correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_validation_loss = validation_loss / num_batches
    validation_losses.append(avg_validation_loss)
    validation_accuracies.append(val_correct / total)
    print(f"Validation Error: \n Accuracy: {(100 * val_correct / total):>0.1f}%, Avg loss: {validation_loss:>8f} \n")


def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size  # Calculate accuracy once after the loop
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    validation_loop(validation_loader, model, loss_fn)
    scheduler.step(round(1 - validation_accuracies[-1], 1))
    print(f"Learning rate after epoch {t + 1}: {scheduler.optimizer.param_groups[0]['lr']} \n \n")
test(test_loader, model, loss_fn)

# Save the trained model
torch.save(model.state_dict(), "trained_gru_model.pth")
print("Model saved successfully.")

#####################

import matplotlib.pyplot as plt

# Plotting
def plot_training_history():
    epoch_numbers = range(1, epochs + 1)

    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epoch_numbers, train_losses, label="Train Loss")
    plt.plot(epoch_numbers, validation_losses, label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epoch_numbers, train_accuracies, label="Train Accuracy")
    plt.plot(epoch_numbers, validation_accuracies, label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call this function after training is complete
plot_training_history()


#####################

# Define the model architecture (same as during training)
model = GRUModel(
    vocab_size=len(vocabulary),
    embedding_dim=100,
    hidden_size=128,
    num_layers=3,
    num_classes=6,
    dropout=0.5,
    freeze_embeddings=False
).to(device)
model.load_state_dict(torch.load("trained_gru_model.pth"))
model.eval()
print("Model loaded successfully.")


# Analyze predictions on the validation set
failure_cases = []

model.eval()  # Ensure the model is in evaluation mode
with torch.no_grad():
    for X, y in validation_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        predicted = pred.argmax(1)  # Predicted class
        for i in range(len(y)):
            if predicted[i] != y[i]:  # Failure case: predicted != true label
                failure_cases.append({
                    "input": X[i].cpu().numpy(),  # Tokenized input
                    "true_label": y[i].item(),
                    "predicted_label": predicted[i].item(),
                    "raw_prediction": pred[i].cpu().numpy()  # Raw prediction scores
                })

# Create a reverse vocabulary mapping
reverse_vocab = {idx: word for word, idx in vocabulary.items()}

# Decode tokens into words
def decode_tokens(tokens):
    return [reverse_vocab[token] for token in tokens if token in reverse_vocab]

# Example:
for case in failure_cases[5:10]:
    decoded_input = decode_tokens(case["input"])
    print("Decoded Input:", " ".join(decoded_input))
    print("True Label:", case["true_label"])
    print("Predicted Label:", case["predicted_label"])
    print("Raw Prediction Scores:", case["raw_prediction"])
    print()


#####################

## Interactive prompt

import torch
import re


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def tokenize_text(text, vocabulary, max_length=25, pad_value=1):
    tokens = text.split()
    token_indices = [vocabulary.get(token, 0) for token in tokens]
    if len(token_indices) > max_length:
        token_indices = token_indices[:max_length]
    else:
        token_indices += [pad_value] * (max_length - len(token_indices))
    print(token_indices)
    return token_indices


def classify_text(model, text, vocabulary, label_mapping):
    preprocessed_text = preprocess_text(text)
    token_indices = tokenize_text(preprocessed_text, vocabulary)
    input_tensor = torch.tensor([token_indices], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
    return label_mapping[predicted_class]


if __name__ == "__main__":
    label_mapping = {
        0: "Sadness",
        1: "Joy",
        2: "Love",
        3: "Anger",
        4: "Fear",
        5: "Surprise"
    }


    while True:
        user_input = input("Enter a text to classify (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        predicted_label = classify_text(model, user_input, vocabulary, label_mapping)
        print(f"Predicted Label: {predicted_label}")