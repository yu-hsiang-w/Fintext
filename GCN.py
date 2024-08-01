import torch
from transformers import BertModel, AutoTokenizer, AutoModel
import random
import os
import json
random.seed(2024)

import numpy as np

import pickle

def check_vectors(v1, v2):
    matching_elements = sum(a == b for a, b in zip(v1, v2))
    return (matching_elements >= 2)


def main():
    with open('para_info_contriever3.pkl', 'rb') as f:
        para_info = pickle.load(f)

    count = 0
    while count < len(para_info):
        if count == 0:
            temp = []
            temp.append(para_info[count][2])
            section_length = (len(para_info)) // 10
            start_index = random.randint(0, len(para_info) - section_length)
            for j in range(start_index, start_index + section_length):
                if check_vectors(para_info[count][0], para_info[j][0]) and (count != j):
                    temp.append(para_info[j][2])
            print(f"Paragraph Number: {count}, Length: {len(temp)}, type 1")
            temp_array = np.array(temp)
            average_array = np.mean(temp_array, axis=0)
            para_info[count][2] = (average_array + para_info[count][2]) / 2
            count = count + 1
        
        else:
            if para_info[count][0] == para_info[count - 1][0]:
                print(f"Paragraph Number: {count}, Length: {len(temp)}, type 2")
                para_info[count][2] = (average_array + para_info[count][2]) / 2
                count = count + 1

            else:
                temp = []
                temp.append(para_info[count][2])
                section_length = (len(para_info)) // 10
                start_index = random.randint(0, len(para_info) - section_length)
                for j in range(start_index, start_index + section_length):
                    if check_vectors(para_info[count][0], para_info[j][0]) and (count != j):
                        temp.append(para_info[j][2])
                print(f"Paragraph Number: {count}, Length: {len(temp)}, type 1")
                temp_array = np.array(temp)
                average_array = np.mean(temp_array, axis=0)
                para_info[count][2] = (average_array + para_info[count][2]) / 2
                count = count + 1

    with open('para_info_GCN3.pkl', 'wb') as f:
        pickle.dump(para_info, f)


main()