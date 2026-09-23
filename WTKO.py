import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

############################################
# Device & Seed
############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(30)
np.random.seed(30)

############################################
# Dataset
############################################
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label

############################################
# One-hot encoding (Pure NumPy - No TensorFlow)
############################################
nucleotides = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}

def one_hot_encode(sequences, seq_length=None):
    encoded_sequences = []
    for seq in sequences:
        seq = seq.upper()
        encoded = np.array([nucleotides.get(b, [0,0,0,0]) for b in seq], dtype=np.float32)
        
        if seq_length:
            current_len = len(encoded)
            if current_len < seq_length:
                # Post-padding with zeros (equivalent to TF's padding='post')
                pad_width = ((0, seq_length - current_len), (0, 0))
                encoded = np.pad(encoded, pad_width, mode='constant', constant_values=0)
            elif current_len > seq_length:
                # Truncate if longer than seq_length
                encoded = encoded[:seq_length]
                
        encoded_sequences.append(encoded)
    return np.array(encoded_sequences)

############################################
# Attention Layer
############################################
class Attention1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = self.attn(x)                # [batch, seq_len, 1]
        weights = torch.softmax(weights, dim=1)
        out = torch.sum(x * weights, dim=1)   # [batch, features]
        return out, weights

############################################
# CNN + Attention Model
############################################
class SequenceModelWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=10)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=10)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.attn = Attention1D(128)
        self.fc1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)                # [batch, channels, seq_len]

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = x.permute(0, 2, 1)                # [batch, seq_len, features]
        x, attn_weights = self.attn(x)

        x = F.relu(self.fc1(x))
        x = self.output(x)

        return x, attn_weights

############################################
# Load Data
############################################
wt_df = pd.read_csv('/home/lopde33/EBF1_Enhancers/WT_NoPromoter512bp.csv', delimiter=";")
ko_df = pd.read_csv('/home/lopde33/EBF1_Enhancers/KO_NoPromoter512bp.csv', delimiter=";")
wt_df['label'] = 1
ko_df['label'] = 0
n = min(len(wt_df), len(ko_df))
df = pd.concat([wt_df.sample(n, random_state=42),
                ko_df.sample(n, random_state=42)]).reset_index(drop=True)
df = df.dropna(subset=['sequence'])

train_df = df[~df['chrom'].isin(['chr6','chr7'])]
test_df = df[df['chrom'].isin(['chr6','chr7'])]

X_train = train_df['sequence'].values
y_train = train_df['label'].values
X_test = test_df['sequence'].values
y_test = test_df['label'].values

train_encoded = one_hot_encode(X_train, seq_length=1000)
test_encoded = one_hot_encode(X_test, seq_length=1000)

train_dataset = SequenceDataset(train_encoded, y_train)
test_dataset = SequenceDataset(test_encoded, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

############################################
# Training Setup
############################################
model = SequenceModelWithAttention().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

epochs = 100
accumulation_steps = 4

train_acc_history = []
test_acc_history = []
best_test_acc = 0.0
best_model_path = "Rela_model.pth"

############################################
# Training Loop
############################################
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    optimizer.zero_grad()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast():
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_acc = correct_train / total_train
    train_acc_history.append(train_acc)

    # Evaluate on test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    test_acc = correct_test / total_test
    test_acc_history.append(test_acc)

    # Save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), best_model_path)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

############################################
# Plot Accuracy vs Epoch
############################################
best_epoch = np.argmax(test_acc_history) + 1
best_test_acc = test_acc_history[best_epoch - 1]

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_acc_history, label="Train Accuracy")
plt.plot(range(1, epochs + 1), test_acc_history, label="Test Accuracy")
plt.scatter(best_epoch, best_test_acc, color="red", label=f"Best {best_test_acc*100:.2f}%")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.legend()
plt.grid()
plt.savefig("Rela_accuracy.png")
plt.show()

############################################
# Final Evaluation & Confusion Matrix
############################################
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["KO", "WT"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Best Model")
plt.savefig("conu_Rela.png")
plt.show()

# Classification Report
print("\nClassification Report (Best Model):\n")
print(classification_report(all_labels, all_preds, target_names=["KO", "WT"]))
