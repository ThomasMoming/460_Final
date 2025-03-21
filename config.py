# 白键（C, D, E, F, G, A, B）对应键盘按键 D F G H J K L
# 黑键（C#, D#, F#, G#, A#）对应键盘按键 R T U I O
# MIDI 60 = C4 (中央C), 61 = C#4, 62 = D4, 63 = D#4, 64 = E4, 65 = F4

# 简谱音阶映射
SCALE_MAP = {
    "C": "1", "D": "2", "E": "3", "F": "4", "G": "5", "A": "6", "B": "7",
    "C#": "1#", "D#": "2#", "F#": "4#", "G#": "5#", "A#": "6#"
}

# 键位映射
KEY_DISPLAY_MAP = {
    "C": "D", "D": "F", "E": "G", "F": "H", "G": "J", "A": "K", "B": "L",
    "C#": "R", "D#": "T", "F#": "U", "G#": "I", "A#": "O"
}

NOTE_MAP = {
    "C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64,
    "F": 65, "F#": 66, "G": 67, "G#": 68, "A": 69,
    "A#": 70, "B": 71
}

KEY_MAP = {
    "d": "C", "f": "D", "g": "E", "h": "F", "j": "G", "k": "A", "l": "B",
    "r": "C#", "t": "D#", "u": "F#", "i": "G#", "o": "A#"
}



# 按键位置
KEY_POSITIONS = {
    "C": (50, 200), "D": (100, 200), "E": (150, 200),
    "F": (200, 200), "G": (250, 200), "A": (300, 200), "B": (350, 200),
    "C#": (75, 150), "D#": (125, 150), "F#": (225, 150),
    "G#": (275, 150), "A#": (325, 150)
}

# 颜色配置
WHITE_KEY_COLOR = "white"
BLACK_KEY_COLOR = "black"
ACTIVE_COLOR = "yellow"


