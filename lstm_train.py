import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os

# ==== 训练设置 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EPOCHS = 30
BATCH_SIZE = 64
SEQUENCE_LENGTH = 50
LEARNING_RATE = 0.001
DATASET_PATH = "dataset.txt"
MODEL_PATH = "lstm_model.pth"
DATA_PATH = "data/"

# ==== 模型结构 ====
class DualLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, note_vocab=128, dur_vocab=256):
        super(DualLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc_note = nn.Linear(hidden_size, note_vocab)
        self.fc_dur = nn.Linear(hidden_size, dur_vocab)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        note_out = self.fc_note(out)
        dur_out = self.fc_dur(out)
        return note_out, dur_out

# ==== 数据处理 ====
def load_dataset():
    sequences, note_targets, dur_targets = [], [], []
    note_set, dur_set = set(), set()

    with open(DATASET_PATH, "r") as f:
        lines = f.readlines()

    notes_durations = []
    for line in lines:
        try:
            pairs = line.strip().split()
            for pair in pairs:
                note, duration = pair.split(":")
                note = int(note)
                duration = int(duration)
                notes_durations.append((note, duration))
                note_set.add(note)
                dur_set.add(duration)
        except:
            continue

    note_to_int = {note: i for i, note in enumerate(sorted(note_set))}
    dur_to_int = {dur: i for i, dur in enumerate(sorted(dur_set))}
    int_to_note = {i: note for note, i in note_to_int.items()}
    int_to_dur = {i: dur for dur, i in dur_to_int.items()}

    os.makedirs(DATA_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, "notes.pkl"), "wb") as f:
        pickle.dump(int_to_note, f)
    with open(os.path.join(DATA_PATH, "durations.pkl"), "wb") as f:
        pickle.dump(int_to_dur, f)

    for i in range(len(notes_durations) - SEQUENCE_LENGTH):
        seq = notes_durations[i:i + SEQUENCE_LENGTH]
        target = notes_durations[i + SEQUENCE_LENGTH]
        seq_norm = [[n / 128.0, d / 1000.0] for n, d in seq]
        sequences.append(seq_norm)
        note_targets.append(note_to_int[target[0]])
        dur_targets.append(dur_to_int[target[1]])

    return sequences, note_targets, dur_targets, len(note_to_int), len(dur_to_int)

class MusicDataset(Dataset):
    def __init__(self, sequences, note_targets, dur_targets):
        self.sequences = sequences
        self.note_targets = note_targets
        self.dur_targets = dur_targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.note_targets[idx], dtype=torch.long),
            torch.tensor(self.dur_targets[idx], dtype=torch.long)
        )

# ==== 训练 ====
def train_network():
    sequences, note_targets, dur_targets, note_vocab, dur_vocab = load_dataset()
    dataset = MusicDataset(sequences, note_targets, dur_targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DualLSTMModel(input_size=2, note_vocab=note_vocab, dur_vocab=dur_vocab).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_note_y, batch_dur_y in dataloader:
            batch_x = batch_x.to(device)
            batch_note_y = batch_note_y.to(device)
            batch_dur_y = batch_dur_y.to(device)

            optimizer.zero_grad()
            note_pred, dur_pred = model(batch_x)
            loss_note = criterion(note_pred, batch_note_y)
            loss_dur = criterion(dur_pred, batch_dur_y)
            loss = loss_note + loss_dur
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("模型训练完成并保存到", MODEL_PATH)

if __name__ == "__main__":
    train_network()