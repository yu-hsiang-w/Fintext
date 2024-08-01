import torch
from transformers import BertModel, AutoTokenizer, AutoModel, DPRContextEncoder
import random
import os
import json
random.seed(2024)

import faiss
import numpy as np

import pickle


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def batch_embed_texts_contriever(texts, model, tokenizer, device='cuda:2', batch_size=512):
    model.to(device)
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            print(i)
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            emb = mean_pooling(model(**inputs)[0], inputs['attention_mask'])
            emb = emb.cpu().numpy()
            embeddings.append(emb)
            #print(len(emb))
    
    return embeddings

def batch_embed_texts_spider(texts, model, tokenizer, device='cuda:2', batch_size=512):
    model.to(device)
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            print(i)
            batch_texts = texts[i:i + batch_size]
            input_dict = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            del input_dict["token_type_ids"]

            outputs = model(**input_dict)
            embeddings.append(outputs[0].cpu().numpy())
            #print(len(outputs[0][0]))
    
    return embeddings


def main():
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    #retriever = AutoModel.from_pretrained('facebook/contriever')
    #retriever = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained("tau/spider")
    retriever = DPRContextEncoder.from_pretrained("tau/spider")

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
    #embeddings = batch_embed_texts_contriever(texts, retriever, tokenizer)
    embeddings = batch_embed_texts_spider(texts, retriever, tokenizer)
    embeddings = np.vstack(embeddings)
    
    para_info = []
    for i in range(len(texts)):
        temp = [position[i], texts[i], embeddings[i], id[i]]
        para_info.append(temp)

    # Save the vector to a file
    with open('para_info_spider_firm.pkl', 'wb') as f:
        pickle.dump(para_info, f)

main()