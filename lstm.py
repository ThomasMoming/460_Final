# lstm.py
import torch
import torch.nn as nn
import numpy as np
import pickle
import time

# === 模型定义 ===
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

# === 全局变量 ===
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

# === 加载模型和映射 ===
def load_lstm_model(
    model_path="lstm_model.pth",
    note_path="data/notes.pkl",
    dur_path="data/durations.pkl"
):
    global model, int_to_note, int_to_dur, note_to_int, dur_to_int, note_vocab, dur_vocab

    with open(note_path, "rb") as f:
        int_to_note = pickle.load(f)
    note_to_int = {v: k for k, v in int_to_note.items()}
    note_vocab = len(note_to_int)

    with open(dur_path, "rb") as f:
        int_to_dur = pickle.load(f)
    dur_to_int = {v: k for k, v in int_to_dur.items()}
    dur_vocab = len(dur_to_int)

    model_obj = DualLSTMModel(input_size=2, note_vocab=note_vocab, dur_vocab=dur_vocab)
    model_obj.load_state_dict(torch.load(model_path, map_location=device))
    model_obj.to(device)
    model_obj.eval()
    model = model_obj
    print("✅ LSTM 模型加载完成")

# === 生成旋律（自动扩展至至少40秒） ===
def generate_lstm_melody(seed_notes, seed_durations, min_duration=40.0, temperature=1.2):
    global generated_melody

    if model is None:
        print("❌ 模型未加载")
        return []

    # 扩展种子序列
    while len(seed_notes) < SEQUENCE_LENGTH:
        seed_notes += seed_notes
        seed_durations += seed_durations
    seed_notes = seed_notes[:SEQUENCE_LENGTH]
    seed_durations = seed_durations[:SEQUENCE_LENGTH]

    melody = []
    total_time = 0.0
    MIN_PITCH = 48  # C3，屏蔽低音

    while total_time < min_duration:
        # 构建输入序列 [note/128.0, dur/1000.0]
        input_seq = [[n / 128.0, d / 1000.0] for n, d in zip(seed_notes, seed_durations)]
        input_tensor = torch.tensor([input_seq], dtype=torch.float32).to(device)

        with torch.no_grad():
            note_pred, dur_pred = model(input_tensor)

            # 采样前屏蔽低音概率（方法2）
            note_probs = torch.softmax(note_pred / temperature, dim=1)
            note_probs[0, :MIN_PITCH] = 0
            note_probs = note_probs / note_probs.sum()

            dur_probs = torch.softmax(dur_pred / temperature, dim=1)

            note_index = torch.multinomial(note_probs, num_samples=1).item()
            dur_index = torch.multinomial(dur_probs, num_samples=1).item()

            pred_note = int(int_to_note[note_index])

            # 强制下限音高（方法1）
            if pred_note < MIN_PITCH:
                pred_note = MIN_PITCH

            pred_dur_ms = int(int_to_dur[dur_index])
            scaled_duration = max(pred_dur_ms * 2 / 1000.0, 0.2)

            melody.append((pred_note, scaled_duration))
            total_time += scaled_duration

        # 更新种子
        seed_notes.append(note_index)
        seed_durations.append(dur_index)
        seed_notes = seed_notes[1:]
        seed_durations = seed_durations[1:]

    print(f"🎼 LSTM 生成旋律（总时长 {total_time:.2f} 秒）")
    generated_melody = melody
    return melody

# === 获取旋律 ===
def get_generated_melody():
    return generated_melody

# === 播放旋律（最多 20 秒） ===
def play_lstm_melody(event_handler):
    melody = get_generated_melody()
    if not melody:
        print("❌ 没有生成的 LSTM 旋律")
        return

    print("▶️ 播放 LSTM 旋律（最多 20 秒）...")
    start_time = time.time()

    for note, duration in melody:
        if time.time() - start_time > 20:
            print("⏹ 播放超时自动停止（20 秒）")
            break
        event_handler.play_midi(note)
        time.sleep(duration)
        event_handler.stop_midi(note)
