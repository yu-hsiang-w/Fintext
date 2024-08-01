import torch
from transformers import BertModel, AutoTokenizer, AutoModel
import random
import os
import json
random.seed(2024)

import faiss
import numpy as np

import pickle

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, k):
    distances, indices = index.search(query_embedding, k)
    return indices, distances

def main():
    with open('para_info_contriever_firm.pkl', 'rb') as f:
        para_info = pickle.load(f)

    # Extract embeddings and texts
    final_embeddings = np.vstack([item[2] for item in para_info]).astype('float32')
    final_texts = [item[1] for item in para_info]
    final_ids = [item[3] for item in para_info]

    # Create a FAISS index
    index = create_faiss_index(final_embeddings)

    #query_text = "Apple Inc. in april 2022, the company announced an increase to its program authorization from $315 billion to $405 billion and raised its quarterly dividend from $0.22 to $0.23 per share beginning in may 2022. during 2022, the company repurchased $90.2 billion of its common stock and paid dividends and dividend equivalents of $14.8 billion."
    #query_text = "Nvidia Recent developments, future objectives and challenges Termination of the arm share purchase agreement on february 8, 2022, nvidia and softbank announced the termination of the share purchase agreement whereby nvidia would have acquired arm from softbank. the parties agreed to terminate because of significant regulatory challenges preventing the completion of the transaction. we intend to record in operating expenses a $1.36 billion charge in the first quarter of fiscal year 2023 reflecting the write-off of the prepayment provided at signing in september 2020."
    #query_text = "Hilton in response to the global crisis resulting from the covid-19 pandemic, we took certain proactive measures in 2020 to help our business withstand the negative impact on our business from the crisis. these measures included securing our liquidity position to be able to meet our obligations for the foreseeable future, including issuing senior notes, drawing down the available borrowing capacity of our $1.75 billion revolving credit facility and consummating the april 2020 pre-sale of hilton honors points to american express for $1.0 billion in cash (the \"honors points pre-sale\"). further, in february 2021, we issued the 3.625% senior notes due 2032 to continue to extend debt maturities and reduce our cost of debt by repaying the outstanding 5.125% senior notes due 2026. based on our continued recovery and expectations of the foreseeable demands on our available cash and our liquidity in future periods, we had fully repaid the outstanding debt balance on the revolving credit facility by june 2021."
    #query_embedding = batch_embed_texts([query_text], retriever, tokenizer)
    #query_id = "20221028_10-K_320193_part2_item7_para7"
    #query_id = "20220318_10-K_1045810_part2_item7_para5"
    #query_id = "20220217_10-K_200406_part2_item7_para42"
    #query_id = "20220216_10-K_1585689_part2_item7_para47"
    #query_id = "20220222_10-K_1090727_part2_item7_para5"
    query_ids = ["20221028_10-K_320193_part2_item7_para7", "20220318_10-K_1045810_part2_item7_para5", "20220217_10-K_200406_part2_item7_para42", "20220216_10-K_1585689_part2_item7_para74", "20220222_10-K_1090727_part2_item7_para5"]
    for query_id in query_ids:
        for i in range(len(final_ids)):
            if final_ids[i] == query_id:
                query_embedding = final_embeddings[i]
                query_text = final_texts[i]
                print("Found!!!!!!")
        
        similar_indices, similar_scores = search_index(index, np.array(query_embedding).reshape(1, -1), k = 11)
        similar_texts = [final_texts[i] for i in similar_indices[0]]
        similar_ids = [final_ids[i] for i in similar_indices[0]]
        

        print("==============================================================")
        print(query_text)
        print("==============================================================")
        print("Similar texts:")
        print()
        
        for i in range(1, len(similar_texts)):
            #print(similar_scores[0][i])
            print(similar_texts[i])
            print()
        
        
        for i in range(1, len(similar_ids)):
            print(f"{query_id} Q0 {similar_ids[i]} {i-1} {similar_scores[0][i]} cnc_alignment")
        
main()