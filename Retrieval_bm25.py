import torch
from transformers import BertModel, AutoTokenizer, AutoModel
import random
import os
import json
random.seed(2024)

import numpy as np

import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def id_to_text(target_id, id_list, text_list):
    for i in range(len(id_list)):
        if id_list[i] == target_id:
            return text_list[i] 

def main():
    with open('para_info_contriever_firm.pkl', 'rb') as f:
        para_info = pickle.load(f)

    # Extract embeddings and texts
    final_texts = [item[1] for item in para_info]
    final_ids = [item[3] for item in para_info]
    
    '''
    tokenized_paragraphs = []
    for i in range(len(final_texts)):
        tokenized_paragraphs.append(word_tokenize(final_texts[i].lower()))
        print(i)

    # Save the vector to a file
    with open('bm25_tokenized_para.pkl', 'wb') as f:
        pickle.dump(tokenized_paragraphs, f)
    '''

    with open('bm25_tokenized_para.pkl', 'rb') as f:
        tokenized_paragraphs = pickle.load(f)

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_paragraphs)
    
    #query_text = "Apple Inc. in april 2022, the company announced an increase to its program authorization from $315 billion to $405 billion and raised its quarterly dividend from $0.22 to $0.23 per share beginning in may 2022. during 2022, the company repurchased $90.2 billion of its common stock and paid dividends and dividend equivalents of $14.8 billion."
    #query_text = "Nvidia Recent developments, future objectives and challenges Termination of the arm share purchase agreement on february 8, 2022, nvidia and softbank announced the termination of the share purchase agreement whereby nvidia would have acquired arm from softbank. the parties agreed to terminate because of significant regulatory challenges preventing the completion of the transaction. we intend to record in operating expenses a $1.36 billion charge in the first quarter of fiscal year 2023 reflecting the write-off of the prepayment provided at signing in september 2020."
    #query_text = "Hilton in response to the global crisis resulting from the covid-19 pandemic, we took certain proactive measures in 2020 to help our business withstand the negative impact on our business from the crisis. these measures included securing our liquidity position to be able to meet our obligations for the foreseeable future, including issuing senior notes, drawing down the available borrowing capacity of our $1.75 billion revolving credit facility and consummating the april 2020 pre-sale of hilton honors points to american express for $1.0 billion in cash (the \"honors points pre-sale\"). further, in february 2021, we issued the 3.625% senior notes due 2032 to continue to extend debt maturities and reduce our cost of debt by repaying the outstanding 5.125% senior notes due 2026. based on our continued recovery and expectations of the foreseeable demands on our available cash and our liquidity in future periods, we had fully repaid the outstanding debt balance on the revolving credit facility by june 2021."
    #query_embedding = batch_embed_texts([query_text], retriever, tokenizer)
    #query_id = "20221028_10-K_320193_part2_item7_para7"
    #query_id = "20220318_10-K_1045810_part2_item7_para5"
    #query_id = "20220217_10-K_200406_part2_item7_para42"
    #query_id = "20220216_10-K_1585689_part2_item7_para47"
    #query_id = "20220222_10-K_1090727_part2_item7_para5"
    query_ids = ["20221028_10-K_320193_part2_item7_para7", "20220318_10-K_1045810_part2_item7_para5", "20220217_10-K_200406_part2_item7_para42", "20220216_10-K_1585689_part2_item7_para47", "20220222_10-K_1090727_part2_item7_para5"]
    for query_id in query_ids:
        query_paragraph = id_to_text(query_id, final_ids, final_texts)
        
        tokenized_query = word_tokenize(query_paragraph.lower())

        assert all(isinstance(p, list) for p in tokenized_paragraphs), "Each paragraph should be a list of tokens."

        # Get scores for the query
        scores = bm25.get_scores(tokenized_query)

        # Rank paragraphs based on scores
        ranked_paragraphs = sorted(zip(final_texts, scores), key=lambda x: x[1], reverse=True)

        # Get top 10 paragraphs
        top_10_paragraphs = ranked_paragraphs[1:11]

        # Print the top 10 paragraphs
        print("==============================================================")
        print("Query text:")
        print(id_to_text(query_id, final_ids, final_texts))
        print("==============================================================")
        print("Similar texts:")
        print()
        for i, (paragraph, score) in enumerate(top_10_paragraphs, start=1):
            print(f"{i}. {paragraph} (Score: {score})")
            print()

        print()
        print()
        print()
        print()
    



main()