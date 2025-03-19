import tkinter as tk
from ui import VirtualPiano
import event_handler
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.PngImagePlugin")

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualPiano(root)
    # 关闭--MIDI
    def on_closing():
        event_handler.close_midi()
        root.destroy()  # 正确关闭 Tkinter 窗口

    root.protocol("WM_DELETE_WINDOW", on_closing)  # 监听窗口关闭事件
    root.mainloop()

