import torch
from transformers import BertModel, AutoTokenizer, AutoModel
import random
import os
import json
random.seed(2024)

import faiss
import numpy as np

import pickle

class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        emb = torch.nn.functional.normalize(emb, dim=-1)

        return emb

def batch_embed_texts(texts, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), batch_size=1080):
    model.to(device)
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            print(i)
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer.batch_encode_plus(
                batch_texts,
                max_length=256,
                truncation=True,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(device)
            
            inputs_tokens, inputs_mask = inputs["input_ids"], inputs["attention_mask"].bool()
            emb = model(input_ids=inputs_tokens, attention_mask=inputs_mask).cpu().numpy()
            embeddings.extend(emb)
    
    return embeddings



def main():
    model_save_path = "/home/yhwang/FinText/Trained_Model"
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    retriever = Contriever.from_pretrained(model_save_path)

    texts = []
    position = []
    id = []

    directory1 = "/home/ythsiao/output"
    firm_list = os.listdir(directory1)

    for firm in firm_list:
        directory2 = os.path.join(directory1, firm)
        directory3 = os.path.join(directory2, "10-K")
        tenK_list = os.listdir(directory3)
        tenK_list = sorted(tenK_list)


        for tenK in tenK_list:
            if tenK[:4] != "2023":
                file_path = os.path.join(directory3, tenK)
                with open(file_path, 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        if 'company_name' in data:
                            firm_name = data['company_name']
                        if 'sector' in data:
                            sector = data['sector']
                        if 'paragraph' in data:
                                concatenated_string = " ".join(data["paragraph"])
                                #text = concatenated_string   #either one of three
                                text = firm_name + " " + concatenated_string  #either one of three
                                #text = sector + " " + concatenated_string #either one of three
                                texts.append(text)

                                id_decompose = data['id'].split('_')
                                year = id_decompose[0][:4]
                                cik = id_decompose[2]
                                item = id_decompose[-2]
                                temp = [year, cik, item]
                                position.append(temp)

                                id.append(data['id'])




    # Embed a list of texts
    embeddings = batch_embed_texts(texts, retriever, tokenizer)
    embeddings = np.vstack(embeddings)
    
    para_info = []
    for i in range(len(texts)):
        temp = [position[i], texts[i], embeddings[i], id[i]]
        para_info.append(temp)

    # Save the vector to a file
    with open('para_info_spider_itrain50.pkl', 'wb') as f:
        pickle.dump(para_info, f)


main()