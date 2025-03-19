import event_handler
import config

class PianoControls:
    def __init__(self, root, keys, display, piano_instance):
        self.keys = keys
        self.display = display
        self.piano_instance = piano_instance

        root.bind("<KeyPress>", self.key_press)
        root.bind("<KeyRelease>", self.key_release)

    def record_and_play(self, key):
        simple_note = config.SCALE_MAP.get(key, key)
        self.display.update_display(simple_note)
        event_handler.highlight_key(self.keys, key, root=None)

    def key_press(self, event):
        if event.char in config.KEY_MAP:
            key = config.KEY_MAP[event.char]
            print(f"键盘输入: {event.char}, 记录的音符: {key}")

            self.piano_instance.record_and_play_foruser(key)

            # 立即高亮
            event_handler.highlight_key(self.keys, key, root=None)

    def key_release(self, event):
        key_char = event.char.lower()
        if key_char in config.KEY_MAP:
            key = config.KEY_MAP[key_char]
            print(f"释放键: {key_char}, 记录的音符: {key}")
            self.piano_instance.key_release(key)
            event_handler.reset_key(self.keys, key)
