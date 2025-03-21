# lstm.py
import numpy as np
import pickle
from keras.models import load_model
import os

# 生成结果缓存
generated_melody = []

# 模型与映射全局变量
model = None
int_to_note = {}
int_to_duration = {}
note_to_int = {}
duration_to_int = {}
n_vocab_note = 0
n_vocab_duration = 0
SEQUENCE_LENGTH = 10

# ---------- 初始化 ----------
def load_lstm_model(
    model_path="weights-improvement-58-1.6636-bigger.hdf5",
    note_path="data/notes",
    duration_path="data/durations"
):
    global model, int_to_note, note_to_int, int_to_duration, duration_to_int, n_vocab_note, n_vocab_duration

    # 加载模型
    if not os.path.exists(model_path):
        print("LSTM 模型文件不存在！")
        return
    model = load_model(model_path)
    print("LSTM 模型已加载。")

    # 加载 note 和 duration 映射
    with open(note_path, "rb") as f:
        notes = pickle.load(f)
    with open(duration_path, "rb") as f:
        durations = pickle.load(f)

    note_names = sorted(set(notes))
    duration_names = sorted(set(durations))

    note_to_int = {note: i for i, note in enumerate(note_names)}
    int_to_note = {i: note for i, note in enumerate(note_names)}

    duration_to_int = {d: i for i, d in enumerate(duration_names)}
    int_to_duration = {i: d for i, d in enumerate(duration_names)}

    n_vocab_note = len(note_names)
    n_vocab_duration = len(duration_names)


# ---------- 生成旋律 ----------
def generate_lstm_melody(seed_notes, seed_durations, length=50):
    global generated_melody

    if model is None:
        print("LSTM 模型未加载，无法生成旋律")
        return []

    # 数据准备
    seed = list(zip(seed_notes, seed_durations))
    if len(seed) < SEQUENCE_LENGTH:
        print(f"种子序列不足 {SEQUENCE_LENGTH} 个，无法生成")
        return []

    # 仅使用最后 SEQUENCE_LENGTH 作为种子
    pattern = seed[-SEQUENCE_LENGTH:]

    # 转为整数索引（使用默认值）
    pattern_notes = [note_to_int.get(str(n), 0) for n, d in pattern]
    pattern_durs = [duration_to_int.get(str(int(d * 4)), 0) for n, d in pattern]  # duration 用四分音符单位

    output = []

    for _ in range(length):
        # reshape 为 LSTM 输入格式 (1, 10, 1)
        input_notes = np.reshape(pattern_notes, (1, SEQUENCE_LENGTH, 1)) / float(n_vocab_note)
        input_durs = np.reshape(pattern_durs, (1, SEQUENCE_LENGTH, 1)) / float(n_vocab_duration)

        # 模型预测
        prediction = model.predict([input_notes, input_durs], verbose=0)
        note_index = np.argmax(prediction[0])
        duration_index = np.argmax(prediction[1])

        result_note = int(int_to_note[note_index])  # 转为 int MIDI 编码
        result_duration = float(int_to_duration[duration_index])  # 四分音符单位

        output.append((result_note, result_duration))

        # 滚动窗口更新
        pattern_notes.append(note_index)
        pattern_notes = pattern_notes[1:]

        pattern_durs.append(duration_index)
        pattern_durs = pattern_durs[1:]

    print("LSTM 生成旋律:", output)
    generated_melody = output
    return output


# ---------- 播放接口 ----------
def get_generated_melody():
    return generated_melody
