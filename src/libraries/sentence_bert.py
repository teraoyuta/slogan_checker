from transformers import BertJapaneseTokenizer, BertModel
import torch

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)
    
    def encode_sentence(self, sentence):
        encoded_input = self.tokenizer.encode_plus(sentence, padding="longest", truncation=True, return_tensors="pt").to(self.device)
        model_output = self.model(**encoded_input)
        sentence_embedding = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

        return sentence_embedding
    
    # def save(self, vecs, file_name):
    #     vecs_list = [vec.tolist() for vec in vecs]
    #     with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
    #         writer = csv.writer(csv_file)
    #         for vec_row in vecs_list:
    #             writer.writerow(vec_row)
    #     return "CSV file saved successfully."
    
    
    def vec_from_list(self, vecs_list):
        vecs_tensor = torch.tensor(vecs_list)
        return vecs_tensor
