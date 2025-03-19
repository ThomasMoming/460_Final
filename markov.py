import markovify

class MarkovMelodyGenerator:
    def __init__(self):
        self.markov_model = None
        self.load_pretrained_model()

    def load_pretrained_model(self):
        """ 从 `data.txt` 训练 Markov Chain 模型 """
        try:
            with open("data.txt", "r", encoding="utf-8") as f:
                data = f.read()
            self.markov_model = markovify.Text(data, state_size=2)  # 训练二阶 Markov Model
            print("Markov Chain 训练完成，模型已加载")
        except Exception as e:
            print(f"训练 Markov 失败: {e}")

    def generate_melody(self, num_notes=15):
        """ 生成新旋律 """
        if not self.markov_model:
            print("需要先训练 Markov Model！")
            return []

        generated_notes = []
        for _ in range(num_notes):
            try:
                note_sequence = self.markov_model.make_sentence()
                if note_sequence:
                    generated_notes.extend(note_sequence.split())
            except:
                break  # 避免生成失败

        print("🎶 Markov 生成的旋律: ", generated_notes)
        return generated_notes  # 只返回音符列表
