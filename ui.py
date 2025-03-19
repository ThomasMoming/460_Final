import tkinter as tk
import config
import event_handler
from piano_display import PianoDisplay
from piano_controls import PianoControls
import time

class VirtualPiano:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Piano")
        self.root.geometry("450x350")
        self.root.configure(bg="gray")
        self.root.focus_force()  # 强制窗口获取键盘焦点

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

        self.root.bind("<Escape>", lambda event: self.stop_recording())

        self.create_buttons()

    def create_buttons(self):
        button_frame = tk.Frame(self.root, bg="gray")
        button_frame.pack(side=tk.BOTTOM, pady=10)

        buttons = [
            ("Start", self.start_recording),
            ("Stop", self.stop_recording),
            ("Markov", self.generate_markov),
            ("Magenta", self.generate_magenta),
            ("Play", self.play_recording)
        ]

        for text, command in buttons:
            btn = tk.Button(button_frame, text=text, font=("Arial", 9, "bold"),
                            width=10, height=1, command=command)
            btn.pack(side=tk.LEFT, padx=5)

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
    def generate_markov(self):
        print("Markov Chain 生成旋律（待实现）")

    def generate_magenta(self):
        print("Magenta AI 生成旋律（待实现）")
