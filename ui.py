import threading
import tkinter as tk
import config
import event_handler
from piano_display import PianoDisplay
from piano_controls import PianoControls
import time
from markov import MarkovMelodyGenerator
from lstm import load_lstm_model, generate_lstm_melody, get_generated_melody, note_to_int, dur_to_int, play_lstm_melody


class VirtualPiano:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Piano")
        self.root.geometry("450x350")
        self.root.configure(bg="gray")
        self.root.focus_force()
        self.trim_dataset()

        self.note_display = PianoDisplay(root)

        self.keys = {}
        for key, (x, y) in config.KEY_POSITIONS.items():
            is_black = "#" in key
            bg_color = config.BLACK_KEY_COLOR if is_black else config.WHITE_KEY_COLOR
            fg_color = "white" if is_black else "black"

            btn = tk.Button(
                root, text=f"{config.SCALE_MAP[key]}\n{config.KEY_DISPLAY_MAP[key]}",
                width=5, height=3, bg=bg_color, fg=fg_color,
                font=("Arial", 10, "bold"),
                command=lambda k=key: self.record_and_play_foruser(k)
            )
            btn.place(x=x, y=y)
            self.keys[key] = btn

        self.controls = PianoControls(self.root, self.keys, self.note_display, self)

        self.recording = False
        self.recorded_notes = []
        self.recorded_timings = []
        self.recorded_durations = {}
        self.active_keys = {}
        self.start_time = None

        # **初始化 Markov 旋律生成器**
        self.markov_generator = MarkovMelodyGenerator()  # 确保这个对象被创建

        # 初始化模型（只执行一次）
        load_lstm_model()

        self.root.bind("<Escape>", lambda event: self.stop_recording())

        self.create_buttons()

    def trim_dataset(self, filename="dataset.txt", max_lines=1361):
        """ 只保留 dataset.txt 文件中的前 max_lines 行 """
        try:
            # 读取文件所有行
            with open(filename, "r") as f:
                lines = f.readlines()

            # 如果文件行数超过 max_lines，则裁剪
            if len(lines) > max_lines:
                with open(filename, "w") as f:
                    f.writelines(lines[:max_lines])  # 只写入前 1361 行
                print(f"dataset.txt 已裁剪到前 {max_lines} 行")
            else:
                print(f"dataset.txt 共有 {len(lines)} 行，无需裁剪")

        except FileNotFoundError:
            print("dataset.txt 文件不存在，跳过裁剪")
        except Exception as e:
            print(f"裁剪 dataset.txt 失败: {e}")

    def create_buttons(self):
        button_frame = tk.Frame(self.root, bg="gray")
        button_frame.pack(side=tk.BOTTOM, pady=10)

        # 设定按钮大小
        button_width = 10
        button_height = 1

        # 创建按钮
        btn_start = tk.Button(button_frame, text="Start", font=("Arial", 9, "bold"),
                              width=button_width, height=button_height, command=self.start_recording)
        btn_stop = tk.Button(button_frame, text="Stop", font=("Arial", 9, "bold"),
                             width=button_width, height=button_height, command=self.stop_recording)
        btn_markov = tk.Button(button_frame, text="Markov", font=("Arial", 9, "bold"),
                               width=button_width, height=button_height, command=self.generate_markov)
        btn_LSTM = tk.Button(button_frame, text="LSTM", font=("Arial", 9, "bold"),
                                width=button_width, height=button_height, command=self.generate_LSTM)
        btn_play = tk.Button(button_frame, text="Play", font=("Arial", 9, "bold"),
                             width=button_width, height=button_height, command=self.play_recording)

        # 播放 Markov 和 LSTM 旋律的按钮
        btn_play_markov = tk.Button(button_frame, text="Play Markov", font=("Arial", 9, "bold"),
                                    width=button_width, height=button_height, command=self.play_markov_melody)
        btn_play_LSTM = tk.Button(button_frame, text="Play LSTM", font=("Arial", 9, "bold"),
                                     width=button_width, height=button_height, command=self.play_LSTM)

        # **使用 Grid 布局，使按钮整齐排列**
        btn_play_markov.grid(row=0, column=2, pady=2)  # Play Markov 在 Markov 按钮正上方
        btn_play_LSTM.grid(row=0, column=3, pady=2)  # Play LSTM 在 LSTM 按钮正上方

        btn_start.grid(row=1, column=0, padx=5, pady=5)
        btn_stop.grid(row=1, column=1, padx=5, pady=5)
        btn_markov.grid(row=1, column=2, padx=5, pady=5)  # Markov 按钮
        btn_LSTM.grid(row=1, column=3, padx=5, pady=5)  # LSTM 按钮
        btn_play.grid(row=1, column=4, padx=5, pady=5)

    def start_recording(self):
        self.recording = True
        self.recorded_notes = []
        self.recorded_timings = []
        self.recorded_durations = {}
        self.active_keys = {}
        self.start_time = None
        self.note_display.clear_display()
        print("开始录制（等待第一个音符）")

    def stop_recording(self):
        self.recording = False
        print("录制完成")
        print(f"最终录制的音符: {self.recorded_notes}")

    def record_and_play_foruser(self, key):
        if key in self.active_keys:
            return

        self.active_keys[key] = time.time()

        if self.recording:
            current_time = time.time()
            if self.start_time is None:
                self.start_time = current_time
                print("录制正式开始")

            timestamp = current_time - self.start_time
            if not self.recorded_timings:
                timestamp = 0

            self.recorded_notes.append(key)
            self.recorded_timings.append(timestamp)
            print(f"录制的音符: {self.recorded_notes}, 时间戳: {self.recorded_timings}")

        event_handler.play_midi(config.NOTE_MAP[key])

        # **确保 Label 正确更新**
        simple_note = config.SCALE_MAP.get(key, key)
        self.note_display.update_display(simple_note)

    def key_release(self, key):
        if key in self.active_keys:
            press_time = self.active_keys.pop(key)
            release_time = time.time()
            duration = release_time - press_time
            self.recorded_durations[key] = duration
            print(f"音符 {key} 持续时间: {duration}")

        event_handler.stop_midi(config.NOTE_MAP[key])

    def play_recording(self):
        if not self.recorded_notes:
            print("没有录制的音符")
            return

        print("播放录制的音符: ", self.recorded_notes)

        for i, note in enumerate(self.recorded_notes):
            if i > 0:
                delay = self.recorded_timings[i] - self.recorded_timings[i - 1]
                time.sleep(max(0, delay))

            duration = self.recorded_durations.get(note, 0.5)
            midi_note = config.NOTE_MAP[note]
            event_handler.play_midi(midi_note)
            time.sleep(duration)
            event_handler.stop_midi(midi_note)

    # **添加这两个方法，避免 `AttributeError`**

    def play_markov_melody(self):
        """ 播放 Markov 生成的旋律，限制 20 秒 或 按 Tab 暂停 """
        if not hasattr(self, 'markov_generator') or not self.markov_generator:
            print("Markov 生成器未初始化，无法播放")
            return

        generated_melody = self.markov_generator.generate_melody(50)  # 生成较长旋律
        if not generated_melody:
            print("无法播放 Markov 旋律")
            return

        self.stop_playing = False  # 允许播放
        self.start_time = time.time()  # 记录播放开始时间

        def play():
            for note, duration in generated_melody:
                if self.stop_playing:
                    print("播放已手动暂停")
                    break

                elapsed_time = time.time() - self.start_time
                if elapsed_time > 20:  # 超过 20 秒自动停止
                    print("播放时间达到 20 秒，自动停止")
                    break

                event_handler.play_midi(note)
                time.sleep(duration / 480.0)  # 将 ticks 转换为秒
                event_handler.stop_midi(note)

            print("播放完成")

        # **使用线程播放，避免 Tkinter 阻塞**
        threading.Thread(target=play, daemon=True).start()

        # 绑定 Tab 键暂停
        self.root.bind("<Tab>", lambda event: self.toggle_playback())

    def toggle_playback(self):
        """ 切换播放状态（按 Tab 键暂停/继续）"""
        self.stop_playing = not self.stop_playing
        if self.stop_playing:
            print("播放已暂停（按 Tab 继续）")
        else:
            print("播放继续")
            self.play_markov_melody()  # 继续播放

    def generate_markov(self):
        """ 将录制的旋律存入 dataset.txt，并更新 Markov 模型 """
        if not self.recorded_notes:
            print("没有录制的音符，无法训练 Markov 模型。")
            return

        try:
            # 组合音符和持续时间，格式：Note:Duration
            new_data = " ".join([f"{config.NOTE_MAP[note]}:{int(self.recorded_durations.get(note, 0.5) * 480)}"
                                 for note in self.recorded_notes])

            # 追加到 dataset.txt
            with open("dataset.txt", "a") as f:
                f.write(new_data + "\n")

            print(f"已存入训练数据: {new_data}")

            # **检查 markov_generator 是否初始化**
            if self.markov_generator:
                self.markov_generator.train_from_file("dataset.txt")
            else:
                print("Markov 生成器未正确初始化")

        except Exception as e:
            print(f"存储训练数据失败: {e}")

    def generate_LSTM(self):
        if not self.recorded_notes:
            print("没有录制的音符，无法生成 LSTM 旋律")
            return

        note_ids = [note_to_int.get(n, 0) for n in self.recorded_notes]
        dur_ids = [dur_to_int.get(int(self.recorded_durations.get(n, 0.5) * 1000), 0) for n in self.recorded_notes]

        generate_lstm_melody(note_ids, dur_ids)

    def play_LSTM(self):
        play_lstm_melody(event_handler)
