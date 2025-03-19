import tkinter as tk

class PianoDisplay:
    def __init__(self, root):
        # 创建音符显示窗口
        self.note_text = tk.StringVar()
        self.notes_history = []  # 记录播放的音符
        self.note_text.set("")  # 初始文本

        self.label = tk.Label(root, textvariable=self.note_text, font=("Arial", 18),
                              bg="gray", fg="white", anchor="center", justify="center", width=30)
        self.label.pack(pady=50)  # 适当向下移动文本框

    def update_display(self, note):
        # 更新音符显示，确保最新音符可见
        MAX_DISPLAY_NOTES = 15
        self.notes_history.append(note)

        if len(self.notes_history) > MAX_DISPLAY_NOTES:
            self.notes_history.pop(0)  # 删除最旧的音符

        print("更新 Label 显示:", " ".join(self.notes_history))  # 调试信息
        self.note_text.set(" ".join(self.notes_history))  # 更新 Label
    def clear_display(self):
        # 清空音符显示
        self.note_text.set("")
        self.notes_history.clear()
