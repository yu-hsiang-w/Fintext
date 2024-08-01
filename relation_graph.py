import torch
from transformers import BertModel, AutoTokenizer, AutoModel
import random
import os
import json
import math
random.seed(2024)

import faiss
import numpy as np

from collections import Counter

import pickle

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, k):
    distances, indices = index.search(query_embedding, k)
    return indices, distances

def softmax(x):
    """Compute the softmax of a list of numbers."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

def main():
    # Load the data
    with open('para_info_contriever_firm.pkl', 'rb') as f:
        para_info = pickle.load(f)

    # Filter the entries to keep only those whose ids start with "2022"
    filtered_para_info = [item for item in para_info if item[3].startswith('2022')]

    # Extract embeddings, texts, and ids from the filtered data
    final_embeddings = np.vstack([item[2] for item in filtered_para_info]).astype('float32')
    final_texts = [item[1] for item in filtered_para_info]
    final_ids = [item[3] for item in filtered_para_info]

    # Create a FAISS index
    index = create_faiss_index(final_embeddings)

    directory1 = "/home/ythsiao/output"
    firm_list = os.listdir(directory1)

    graph_list = []

    for firm in firm_list:
        directory2 = os.path.join(directory1, firm)
        directory3 = os.path.join(directory2, "10-K")
        tenK_list = os.listdir(directory3)
        tenK_list = sorted(tenK_list)


        for tenK in tenK_list:
            if tenK[:4] == "2022":
                file_path = os.path.join(directory3, tenK)
                with open(file_path, 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        if 'cik' in data:
                            cik = data['cik']
                        if 'paragraph' in data:
                            if cik not in graph_list:
                                graph_list.append(cik)

    print(len(graph_list))
    # If you want to initialize the inner values with something specific (e.g., 0), you can do this:
    graph_dict = {key: {sub_key: 0 for sub_key in graph_list} for key in graph_list}
    
    total_number_of_related = 0
    count = 0
    for firm in firm_list:
        directory2 = os.path.join(directory1, firm)
        directory3 = os.path.join(directory2, "10-K")
        tenK_list = os.listdir(directory3)
        tenK_list = sorted(tenK_list)
        for tenK in tenK_list:
            if tenK.startswith("2022"):
                tenK_split = tenK.split('.')[0]
                firm_code = tenK_split.split('_')[2]
                count = count + 1
                print(f'({count}): {tenK_split}')

                base_strings = [f'{tenK_split}_part1_item1_para', f'{tenK_split}_1045810_part1_item1a_para', f'{tenK_split}_part2_item7a_para']
                firm_dict = {}

                for base_string in base_strings:
                    start = 1  # starting point of the iteration
                    end = 1000   # end point of the iteration (inclusive)

                    query_ids = []

                    for i in range(start, end + 1):
                        new_string = f"{base_string}{i}"
                        query_ids.append(new_string)

                    for query_id in query_ids:
                        check = 0
                        for i in range(len(final_ids)):
                            if final_ids[i] == query_id:
                                check = 1
                                query_embedding = final_embeddings[i]
                                query_text = final_texts[i]
                                break

                        if check == 0:
                            break
                        else:
                            similar_indices, similar_scores = search_index(index, np.array(query_embedding).reshape(1, -1), k = 11)
                            similar_texts = [final_texts[i] for i in similar_indices[0]]
                            similar_ids = [final_ids[i] for i in similar_indices[0]]
                            
                            for i in range(1, len(similar_texts)):
                                #print(round(similar_scores[0][i], 4))
                                #print(similar_texts[i])
                                if (firm_code != similar_ids[i].split('_')[2]) and (similar_ids[i].startswith("2022")):
                                    #print(f'{query_id}, {similar_ids[i]}')
                                    parts = similar_ids[i].split('_')
                                    #print(f'{parts[2]}, {similar_scores[0][i]}')
                                    graph_dict[firm_code][parts[2]] = graph_dict[firm_code][parts[2]] + similar_scores[0][i]
        
        for value in graph_dict.values():
            values_list = list(value.values())
            row_sum = sum(values_list)
            print(f"{values_list} Sum: {row_sum}")                            
        
        #if count == 3:
            #break

    for key, inner_dict in graph_dict.items():
        # Extract non-zero values
        non_zero_values = {k: v for k, v in inner_dict.items() if v != 0}
        total = sum(non_zero_values.values())
        if total > 0:
            proportion_values = {k: v / total for k, v in non_zero_values.items()}
            # Replace original values with proportional values, keeping zeros intact
            for k in inner_dict:
                if inner_dict[k] != 0:
                    inner_dict[k] = proportion_values[k]
    
    with open('firms_relation_graph_2022.json', 'w') as json_file:
        json.dump(graph_dict, json_file, indent=4)

                        

main()