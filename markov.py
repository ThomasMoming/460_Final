import markovify

class MarkovMelodyGenerator:
    def __init__(self):
        self.markov_model = None
        self.ticks_per_second = 480  # MIDI 标准
        self.train_from_file("dataset.txt")  # 初始化时加载数据

    def train_from_file(self, dataset_path):
        """ 从 dataset.txt 训练 Markov Chain """
        try:
            with open(dataset_path, "r") as f:
                data = f.read()
            self.markov_model = markovify.NewlineText(data)  # 训练 Markov 模型
            print("Markov 模型已更新！")
        except Exception as e:
            print(f"加载训练数据失败: {e}")

    def generate_melody(self, length=10):
        """ 生成旋律，返回格式 [("note", duration), ...] """
        if not self.markov_model:
            print("Markov 模型未训练，无法生成旋律。")
            return []

        try:
            generated_sequence = self.markov_model.make_sentence()
            if not generated_sequence:
                print("生成失败，数据不足")
                return []

            melody = []
            for pair in generated_sequence.split(" "):
                note, duration = pair.split(":")
                melody.append((int(note), int(duration)))

            print("生成的 Markov 旋律:", melody)
            return melody

        except Exception as e:
            print(f"生成旋律失败: {e}")
            return []
