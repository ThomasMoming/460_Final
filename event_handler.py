import config
import pygame.midi

# 初始化 MIDI 设备
pygame.midi.init()
midi_out = None  # 先初始化为空

try:
    if pygame.midi.get_count() > 0:  # 确保有可用的 MIDI 设备
        midi_out = pygame.midi.Output(0)
        midi_out.set_instrument(0)  # 0 代表钢琴
    else:
        print("没有可用的 MIDI 设备，MIDI 播放将被禁用")
except pygame.midi.MidiException:
    print("MIDI 初始化失败")
    midi_out = None



# 记录音符，避免重复触发
active_notes = set()

def play_midi(note):
    # 播放对应的 MIDI 音符
    if midi_out and note:
        if note in active_notes:
            stop_midi(note)  # 先停止，再重新播放
        midi_out.note_on(note, 127)
        active_notes.add(note)


def stop_midi(note):
    # 停止播放音符
    if midi_out and note in active_notes:
        midi_out.note_off(note, 127)
        active_notes.remove(note)



def highlight_key(keys, key, root):
    # 高亮按键并播放
    if key in keys:
        print(f"高亮按键: {key}")  # 调试信息
        keys[key].config(bg=config.ACTIVE_COLOR)
        keys[key].update_idletasks()
        midi_note = config.NOTE_MAP.get(key)
        play_midi(midi_note)


def reset_key(keys, key):
    # 恢复默认并停止
    if key in keys:
        default_color = config.WHITE_KEY_COLOR if "#" not in key else config.BLACK_KEY_COLOR
        keys[key].after(100, lambda: keys[key].config(bg=default_color))  # 延迟 100ms
        print(f"取消高亮: {key}")  # 调试信息
        midi_note = config.NOTE_MAP.get(key)
        stop_midi(midi_note)  # 停止 MIDI 音符

def key_press(event, keys, root):
    # 监听键盘按下
    if event.char in config.KEY_MAP:
        highlight_key(keys, config.KEY_MAP[event.char], root)

def key_release(event, keys):
    # 监听键盘松开
    if event.char in config.KEY_MAP:
        reset_key(keys, config.KEY_MAP[event.char])

# 退出时关闭 MIDI
def close_midi():
    """ 关闭 MIDI 设备 """
    global midi_out  # 确保修改全局变量
    if midi_out:
        print("关闭 MIDI 输出设备...")
        midi_out.close()
        midi_out = None  # 避免多次关闭
    if pygame.midi.get_init():  # 只有在初始化的情况下才退出
        pygame.midi.quit()
    print("MIDI 已关闭")