import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# **超参数**
EPOCHS = 200  # 训练轮数
BATCH_SIZE = 64  # 批量大小
SEQUENCE_LENGTH = 100  # LSTM 输入的序列长度
LEARNING_RATE = 0.001  # 学习率
MIDI_DIR = "Classical/**/*.mid"  # **递归搜索所有子目录**
MODEL_PATH = "lstm_model.pth"  # **模型保存路径**
DATA_PATH = "data/"  # **数据存储目录**

# **检查 GPU**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# **LSTM 模型**
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, output_size=128):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步
        out = self.fc(out)
        return out


# **提取 MIDI 音符**
def get_notes():
    """ 从 `Classical/` 目录中提取所有音符、offset 和 duration"""
    notes, offsets, durations = [], [], []

    for file in glob.glob(MIDI_DIR, recursive=True):
        print(f"解析: {file}")
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"跳过文件（无法解析）: {file} - 错误: {e}")
            continue  # **跳过该文件**

        try:
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
        except Exception:
            notes_to_parse = midi.flat.notes

        offset_base = 0
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                offsets.append(str(element.offset - offset_base))
                durations.append(str(element.duration.quarterLength))
                offset_base = element.offset
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                offsets.append(str(element.offset - offset_base))
                durations.append(str(element.duration.quarterLength))
                offset_base = element.offset

    # **保存数据**
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(DATA_PATH + "notes.pkl", "wb") as f:
        pickle.dump(notes, f)
    with open(DATA_PATH + "durations.pkl", "wb") as f:
        pickle.dump(durations, f)

    return notes, offsets, durations


# **数据集定义**
class MusicDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)


# **准备数据**
def prepare_sequences(notes, n_vocab):
    """ 将音符转换为 LSTM 可用的序列 """
    note_to_int = {note: number for number, note in enumerate(sorted(set(notes)))}
    sequences, targets = [], []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        sequences.append([note_to_int[n] for n in sequence_in])
        targets.append(note_to_int[sequence_out])

    return np.array(sequences) / float(n_vocab), np.array(targets)


# **训练模型**
def train_network():
    """ 训练 LSTM 网络 """
    notes, offsets, durations = get_notes()
    n_vocab_notes = len(set(notes))

    #  生成训练数据
    sequences, targets = prepare_sequences(notes, n_vocab_notes)
    dataset = MusicDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # **创建 LSTM 模型**
    model = LSTMModel(input_size=1, output_size=n_vocab_notes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # **训练循环**
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = batch_x.unsqueeze(-1)  # **增加最后一维**

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(dataloader):.4f}")

        # **保存最优模型**
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"weights-improvement-{epoch + 1}.pth")

    # **保存最终模型**
    torch.save(model.state_dict(), MODEL_PATH)
    print("训练完成，模型已保存！")


if __name__ == '__main__':
    train_network()
