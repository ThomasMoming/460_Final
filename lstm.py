import torch
import torch.nn as nn
import numpy as np
import pickle
import time
import sys
import os

# 支持 PyInstaller 打包资源路径
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 模型结构定义
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

# 全局变量
model = None
int_to_note = {}
int_to_dur = {}
note_to_int = {}
dur_to_int = {}
note_vocab = 0
dur_vocab = 0
SEQUENCE_LENGTH = 50
generated_melody = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型与映射
def load_lstm_model(
    model_path="lstm_model.pth",
    note_path="data/notes.pkl",
    dur_path="data/durations.pkl"
):
    global model, int_to_note, int_to_dur, note_to_int, dur_to_int, note_vocab, dur_vocab

    with open(resource_path(note_path), "rb") as f:
        int_to_note = pickle.load(f)
    note_to_int = {v: k for k, v in int_to_note.items()}
    note_vocab = len(note_to_int)

    with open(resource_path(dur_path), "rb") as f:
        int_to_dur = pickle.load(f)
    dur_to_int = {v: k for k, v in int_to_dur.items()}
    dur_vocab = len(dur_to_int)

    model_obj = DualLSTMModel(input_size=2, note_vocab=note_vocab, dur_vocab=dur_vocab)
    model_obj.load_state_dict(torch.load(resource_path(model_path), map_location=device))
    model_obj.to(device)
    model_obj.eval()
    model = model_obj
    print("LSTM 模型加载完成")

# 生成旋律（控制在 C4-C5）
def generate_lstm_melody(seed_notes, seed_durations, min_duration=40.0, temperature=1.2):
    global generated_melody

    if model is None:
        print("模型未加载")
        return []

    while len(seed_notes) < SEQUENCE_LENGTH:
        seed_notes += seed_notes
        seed_durations += seed_durations
    seed_notes = seed_notes[:SEQUENCE_LENGTH]
    seed_durations = seed_durations[:SEQUENCE_LENGTH]

    melody = []
    total_time = 0.0

    MIN_PITCH = 48
    MAX_PITCH = 84
    C_MAJOR = {60, 62, 64, 65, 67, 69, 71, 72}

    rhythm_template = [0.25, 0.5, 0.5, 1.0, 1.0, 2.0]
    rhythm_index = 0

    while total_time < min_duration:
        input_seq = [[n / 128.0, d / 1000.0] for n, d in zip(seed_notes, seed_durations)]
        input_tensor = torch.tensor([input_seq], dtype=torch.float32).to(device)

        with torch.no_grad():
            note_pred, dur_pred = model(input_tensor)
            note_probs = torch.softmax(note_pred / temperature, dim=1)
            dur_probs = torch.softmax(dur_pred / temperature, dim=1)

            note_probs[0, :MIN_PITCH] = 0
            note_probs[0, MAX_PITCH + 1:] = 0
            note_probs = note_probs / note_probs.sum()

            note_index = torch.multinomial(note_probs, num_samples=1).item()
            dur_index = torch.multinomial(dur_probs, num_samples=1).item()

            pred_note = int(int_to_note[note_index])
            pred_dur_sec = rhythm_template[rhythm_index % len(rhythm_template)]
            rhythm_index += 1

            last_note = seed_notes[-1]
            if abs(pred_note - last_note) > 12:
                pred_note = last_note
            if pred_note not in C_MAJOR:
                pred_note = last_note
            if len(melody) >= 2 and pred_note == melody[-1][0] == melody[-2][0]:
                pred_note = last_note

            melody.append((pred_note, pred_dur_sec))
            total_time += pred_dur_sec

            seed_notes.append(note_index)
            seed_durations.append(dur_index)
            seed_notes = seed_notes[1:]
            seed_durations = seed_durations[1:]

    print(f"LSTM 生成旋律（总时长 {total_time:.2f} 秒）")
    generated_melody = melody
    return melody

# 获取旋律
def get_generated_melody():
    return generated_melody

# 播放旋律
def play_lstm_melody(event_handler):
    melody = get_generated_melody()
    if not melody:
        print("没有生成的 LSTM 旋律")
        return

    print("播放 LSTM 旋律（最多 20 秒）...")
    start_time = time.time()

    for note, duration in melody:
        if time.time() - start_time > 20:
            print("播放超时自动停止（20 秒）")
            break
        event_handler.play_midi(note)
        time.sleep(duration)
        event_handler.stop_midi(note)
