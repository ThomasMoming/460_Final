import markovify

class MarkovMelodyGenerator:
    def __init__(self):
        self.markov_model = None
        self.ticks_per_second = 480  # 假设 480 ticks = 1 秒（MIDI 标准值）
        self.load_pretrained_model()

    def load_pretrained_model(self):
        """ 从 `dataset.txt` 训练 Markov Chain 模型，支持音符+时间 """
        try:
            with open("dataset.txt", "r", encoding="utf-8") as f:
                data = f.readlines()

            training_data = []
            for line in data:
                if line.strip() and not line.endswith(":\n"):  # 跳过空行和文件名
                    training_data.append(line.strip())

            if not training_data:
                raise ValueError("数据集为空，无法训练 Markov Model！")

            self.markov_model = markovify.Text("\n".join(training_data), state_size=2)  # 训练二阶 Markov Model
            print("Markov Chain 训练完成，模型已加载")
        except Exception as e:
            print(f"训练 Markov 失败: {e}")

    def generate_melody(self, target_duration=30):
        """ 生成旋律，使总时长接近 30s """
        if not self.markov_model:
            print("需要先训练 Markov Model")
            return []

        generated_melody = []
        total_ticks = 0
        max_ticks = target_duration * self.ticks_per_second  # 计算 30s 目标 tick 数

        while total_ticks < max_ticks:
            try:
                note_sequence = self.markov_model.make_sentence()  # 生成一段旋律
                if note_sequence:
                    notes = note_sequence.split("\n")  # 处理 Markov 生成的数据格式
                    for note in notes:
                        if "," in note:
                            note_name, ticks = note.split(",")
                            try:
                                ticks = int(ticks)  # 确保时间戳为整数
                                if total_ticks + ticks > max_ticks:
                                    break  # 避免超过 30 秒
                                generated_melody.append((int(note_name), ticks))
                                total_ticks += ticks
                            except ValueError:
                                continue  # 忽略格式错误的数据
            except:
                break  # 避免异常导致死循环

        print(f" Markov 生成的旋律（{total_ticks / self.ticks_per_second:.2f} 秒）: ", generated_melody)
        return generated_melody  # 返回音符 + 持续时间的列表
