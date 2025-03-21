import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os

# **检查 GPU**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == torch.device("cpu"):
    print("没有检测到可用的 GPU，训练终止。请在 GPU 设备上运行此代码。")
    exit(1)

print(f"训练将在 {device} 上进行 ")

# **超参数**
EPOCHS = 200  # 训练轮数
BATCH_SIZE = 64  # 批量大小
SEQUENCE_LENGTH = 100  # LSTM 输入的序列长度
LEARNING_RATE = 0.001  # 学习率
DATASET_PATH = "dataset.txt"  # 训练数据文件
MODEL_PATH = "lstm_model.pth"  # 模型保存路径
DATA_PATH = "data/"  # 数据存储目录


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


# **读取 dataset.txt**
def load_dataset():
    """ 从 dataset.txt 读取音符和持续时间 """
    notes, durations = [], []

    with open(DATASET_PATH, "r") as f:
        lines = f.readlines()

    for line in lines:
        pairs = line.strip().split(" ")
        for pair in pairs:
            try:
                note, duration = pair.split(":")
                notes.append(int(note))  # 转换为整数
                durations.append(int(duration))
            except ValueError:
                print(f"跳过无效数据: {pair}")

    # **保存音符映射**
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(DATA_PATH + "notes.pkl", "wb") as f:
        pickle.dump(notes, f)

    return notes


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
def prepare_sequences(notes):
    """ 将音符转换为 LSTM 可用的序列 """
    note_to_int = {note: number for number, note in enumerate(sorted(set(notes)))}
    sequences, targets = [], []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        sequences.append([note_to_int[n] for n in sequence_in])
        targets.append(note_to_int[sequence_out])

    return np.array(sequences) / float(len(note_to_int)), np.array(targets), note_to_int


# **训练模型**
def train_network():
    """ 训练 LSTM 网络 """
    notes = load_dataset()
    sequences, targets, note_to_int = prepare_sequences(notes)
    dataset = MusicDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # **创建 LSTM 模型**
    model = LSTMModel(input_size=1, output_size=len(note_to_int)).to(device)
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
    print("🎉 训练完成，模型已保存！")


if __name__ == '__main__':
    train_network()
