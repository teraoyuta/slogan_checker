from libraries.sentence_bert import SentenceBertJapanese
from api.models import slogans
import torch.nn.functional as F

class SloganService:
    def __init__(self):
        self.sentence_bert_model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens-v2")

    def get_sentence_distance(self, sentence):
        all_slogans = slogans.objects.all()
        vector_list = [[float(value) for value in slogan.vector.split(',')] for slogan in all_slogans]
        slogan_list = [slogan.slogan_sentence for slogan in all_slogans]
        vecs = self.sentence_bert_model.vec_from_list(vector_list)
        target_vec = self.sentence_bert_model.encode_sentence(sentence)

        # コサイン類似度による類似度算出
        distance_list = F.cosine_similarity(target_vec, vecs).tolist()

        json_data = []
        # name_listとage_listをzipして辞書のリストを作成する
        for slogan_kana, distance in zip(slogan_list, distance_list):
            entry = {
                "slogan_kana": slogan_kana,
                "distance": round(distance, 2),
            }
            json_data.append(entry)
        sorted_json_data = sorted(json_data, key=lambda x: x['distance'], reverse=True)
        return sorted_json_data
