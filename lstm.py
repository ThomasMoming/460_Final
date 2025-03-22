# lstm.py
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import time

# 模型定义（结构要与训练时保持一致）
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, output_size=128):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# 全局变量
model = None
note_to_int = {}
int_to_note = {}
n_vocab = 0
generated_melody = []
SEQUENCE_LENGTH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和映射
def load_lstm_model(
    model_path="lstm_model.pth",
    note_path="data/notes.pkl"
):
    global model, note_to_int, int_to_note, n_vocab

    # 读取 note 数据
    with open(note_path, "rb") as f:
        notes = pickle.load(f)

    note_names = sorted(set(notes))
    note_to_int = {note: i for i, note in enumerate(note_names)}
    int_to_note = {i: note for i, note in enumerate(note_names)}
    n_vocab = len(note_names)

    # 构建模型
    model_obj = LSTMModel(input_size=1, output_size=n_vocab)
    model_obj.load_state_dict(torch.load(model_path, map_location=device))
    model_obj.to(device)
    model_obj.eval()

    model = model_obj
    print("LSTM 模型加载完成")

# 生成旋律
def generate_lstm_melody(seed_notes, durations, length=50):
    global generated_melody

    if model is None:
        print("模型未加载")
        return []

    if len(seed_notes) < SEQUENCE_LENGTH:
        print("种子序列不足 100 个音符，正在扩展...")
        while len(seed_notes) < SEQUENCE_LENGTH:
            seed_notes += seed_notes  # 重复添加
        seed_notes = seed_notes[:SEQUENCE_LENGTH]  # 截断刚好 100

    # 转为索引并归一化
    pattern = seed_notes[-SEQUENCE_LENGTH:]
    pattern_idx = [note_to_int.get(n, 0) for n in pattern]
    output = []

    for _ in range(length):
        input_seq = np.reshape(pattern_idx, (1, SEQUENCE_LENGTH, 1)) / float(n_vocab)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)
            index = torch.argmax(prediction).item()
            result_note = int(int_to_note[index])
            output.append((result_note, 0.5))  # 默认每个持续 0.5 秒

            # 更新种子
            pattern_idx.append(index)
            pattern_idx = pattern_idx[1:]

    print("LSTM 生成旋律:", output)
    generated_melody = output
    return output

# 获取生成结果
def get_generated_melody():
    return generated_melody
