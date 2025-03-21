import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation, concatenate, Input
import tensorflow.keras.utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import os

# 递归搜索 `Classical/` 目录中的所有 `.mid` 文件
MIDI_DIR = "Classical/**/*.mid"  # **递归搜索所有子目录**
DATA_DIR = "data/"

# 训练超参数
EPOCHS = 200  # 训练轮数（可以调整）
BATCH_SIZE = 64  # 批量大小
SEQUENCE_LENGTH = 100  # LSTM 输入的序列长度


def get_notes():
    """ 递归搜索 Classical/ 目录，提取所有音符和持续时间 """
    notes, offsets, durations = [], [], []

    for file in glob.glob(MIDI_DIR, recursive=True):
        print(f"解析: {file}")
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"跳过文件（无法解析）: {file} - 错误: {e}")
            continue  # **跳过该文件**

        try:
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
        except Exception:
            notes_to_parse = midi.flat.notes

        offset_base = 0
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                offsets.append(str(element.offset - offset_base))
                durations.append(str(element.duration.quarterLength))
                offset_base = element.offset
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                offsets.append(str(element.offset - offset_base))
                durations.append(str(element.duration.quarterLength))
                offset_base = element.offset

    return notes, offsets, durations


def prepare_sequences(notes, n_vocab):
    """ 将音符转换为 LSTM 可用的序列 """
    note_to_int = {note: number for number, note in enumerate(sorted(set(notes)))}
    network_input, network_output = [], []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[n] for n in sequence_in])
        network_output.append(note_to_int[sequence_out])

    network_input = np.reshape(network_input, (len(network_input), SEQUENCE_LENGTH, 1)) / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_network(input_shape, n_vocab):
    """ 创建 LSTM 模型 """
    input_layer = Input(shape=(input_shape[1], input_shape[2]))

    x = LSTM(256, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(512)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(n_vocab, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    return model


def train_network():
    """ 训练 LSTM 网络 """
    notes, offsets, durations = get_notes()

    # 计算音符类别数
    n_vocab_notes = len(set(notes))

    #  生成训练数据
    network_input, network_output = prepare_sequences(notes, n_vocab_notes)

    #  构建 LSTM 模型
    model = create_network(network_input.shape, n_vocab_notes)

    #  定义模型保存路径
    checkpoint_filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    #  训练模型
    model.fit(network_input, network_output, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list, verbose=1)

    #  保存最终模型
    model.save("final_model.hdf5")


if __name__ == '__main__':
    train_network()
