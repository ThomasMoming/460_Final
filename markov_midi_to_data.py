import os
import mido

# 设置文件夹的相对路径
midi_folder = os.path.join(os.getcwd(), "Classical")  # 读取 Classical 目录
dataset_path = os.path.join(os.getcwd(), "dataset.txt")  # 结果保存到 460_Final 目录

def extract_midi_data(midi_folder):
    """
    递归读取 Classical 文件夹中的所有 MIDI 文件，提取音符和持续时间。
    """
    if not os.path.exists(midi_folder):
        print(f"错误：文件夹 '{midi_folder}' 不存在！")
        return []

    midi_data = []  # 存储所有 MIDI 文件的音符序列
    found_midi = False  # 标志位，检查是否找到 MIDI 文件

    for root, _, files in os.walk(midi_folder):  # 递归遍历所有子文件夹
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                found_midi = True
                midi_path = os.path.join(root, file)
                print(f"处理 MIDI 文件: {midi_path}")

                try:
                    midi = mido.MidiFile(midi_path)
                except Exception as e:
                    print(f"无法解析 {file}，错误：{e}")
                    continue  # 跳过无法解析的 MIDI 文件

                note_sequence = []  # 存储单个 MIDI 文件的音符时间序列
                note_start_times = {}  # 记录每个音符的起始时间
                current_time = 0  # 绝对时间

                for track in midi.tracks:
                    for msg in track:
                        current_time += msg.time  # 更新当前时间

                        if msg.type == 'note_on' and msg.velocity > 0:
                            note_start_times[msg.note] = current_time  # 记录音符开始时间
                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                            if msg.note in note_start_times:
                                duration = current_time - note_start_times[msg.note]  # 计算持续时间
                                note_sequence.append(f"{msg.note}:{duration}")  # 存储 音符:持续时间
                                del note_start_times[msg.note]  # 移除已处理的音符

                if note_sequence:
                    midi_data.append(" ".join(note_sequence) + "\n")  # 以空格连接

    if not found_midi:
        print(f"错误：'{midi_folder}' 目录及其子目录中没有 MIDI 文件！")

    return midi_data

def save_dataset(data, save_path):
    """
    将处理后的 MIDI 数据集保存到文本文件中，以便用于 Markov Chain 训练。
    """
    if not data:
        print("数据集为空，未保存 dataset.txt。")
        return

    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(data)

    print(f"数据集已成功保存到 {save_path}")

# 读取 MIDI 数据并保存
midi_data = extract_midi_data(midi_folder)
save_dataset(midi_data, dataset_path)
