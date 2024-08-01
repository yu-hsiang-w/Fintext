import torch
from transformers import BertModel, AutoTokenizer, AutoModel
import random
import os
import json
random.seed(2024)

import faiss
import numpy as np

import pickle
import re
from collections import Counter


import re

from collections import Counter

def unique_words(strings):
    # Count occurrences of each word across all strings
    word_counts = {}
    for string in strings:
        for word in set(string):
            word_counts[word] = word_counts.get(word, 0) + 1

    # Find unique words in each string
    unique_words_per_string = []
    for string in strings:
        unique_words = [word for word in string if word_counts[word] == 2]
        unique_words_per_string.append(unique_words)

    return unique_words_per_string


def filter_strings(strings):
    filtered_strings = []
    for string in strings:
        word_counts = {}
        for word in string:
            word_counts[word] = word_counts.get(word, 0) + 1
        filtered_string = list(set([word for word in string if word_counts[word] >= 10]))
        filtered_strings.append(filtered_string)
    return filtered_strings

def split_into_pairs_with_overlap(text):
    chunk_length = 1
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_length]) for i in range(len(words) - chunk_length + 1)]
    return chunks

def remove_non_alphabetic(data):
    cleaned_data = [[s for s in sublist if s.isalpha()] for sublist in data]
    return cleaned_data


def main():

    task = input("Task type: ")
    task = int(task)

    directory1 = "/home/ythsiao/output"
    firm_list = os.listdir(directory1)
    existed_dataset = []
    firms_string = {}
    sectors_string = {}
    sector_firm = {}
    string_split = {}

    total_count = 0

    for firm in firm_list:
        total_count = total_count + 1
        print(firm)
        sector = ""
        firm_name = ""
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
                        if 'sector' in data:
                            sector = data["sector"]
                        if 'company_name' in data:
                            firm_name = data["company_name"]
                        
                        if sector not in sector_firm:
                            sector_firm[sector] = [firm_name]
                        elif sector in sector_firm:
                            sector_firm[sector].append(firm_name)

                        if 'paragraph' in data:
                                concatenated_string = " ".join(data["paragraph"])                   
                                
                                if sector in sectors_string:
                                    sectors_string[sector].append(concatenated_string)
                                elif sector not in sectors_string:
                                    sectors_string[sector] = [concatenated_string]

                                if firm_name in firms_string:
                                    firms_string[firm_name].append(concatenated_string)
                                elif firm_name not in firms_string:
                                    firms_string[firm_name] = [concatenated_string]

                                if concatenated_string not in string_split:
                                    string_split[concatenated_string] = concatenated_string.split()
                        

    for key, value in sector_firm.items():
        sector_firm[key] = list(set(sector_firm[key]))
    
    for key, value in firms_string.items():
        firms_string[key] = split_into_pairs_with_overlap(" ".join(firms_string[key]))
    print(firms_string["Nvidia Corp"])

    if task == 1:
        strings = []
        for key, value in firms_string.items():
            strings.append(value)
        
        result = unique_words(strings)
        #print(result[-1])
        result = filter_strings(result)
        #print(result[-1])
        result = remove_non_alphabetic(result)

        i = 0
        for key, value in firms_string.items():
            firms_string[key] = result[i]
            i = i + 1

        for key, value in firms_string.items():
            for word in value:
                print()
                print()
                print()
                print(word)
                for firm in firm_list:
                    found = False  # Flag to indicate if the word is found
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
                                        firm_name = data["company_name"]

                                    if 'paragraph' in data:
                                        concatenated_string = " ".join(data["paragraph"])
                                        if word in string_split[concatenated_string]:
                                            print(firm_name)
                                            print(concatenated_string)
                                            print()
                                            found = True  # Set the flag to True
                                            break  # Exit the innermost loop

                            if found:
                                break  # Exit the loop over tenK_list



    
    
    
    
    
    if task == 2:
        for key, value in sectors_string.items():
            sectors_string[key] = split_into_pairs_with_overlap(" ".join(sectors_string[key]))
        #print(sectors_string['Materials'])

        strings = []
        for key, value in sectors_string.items():
            strings.append(value)
        
        result = unique_words(strings)
        #print(result[-1])
        result = filter_strings(result)
        #print(result[-1])

        i = 0
        for key, value in sectors_string.items():
            sectors_string[key] = result[i]
            i = i + 1

        for key, value in sectors_string.items():
            print(key)
            print(value)
            print()
        
        
        # Convert firms_string values to sets for faster lookup
        firms_string_sets = {firm: set(firms_string[firm]) for firm in firms_string}
        print(firms_string_sets["Nvidia Corp"])
        
        final_keyword = {}
        for key, words in sectors_string.items():
            print(key)
            final_keyword[key] = []
            for word in words:
                count = 0
                for firm in sector_firm[key]:
                    if firm in firms_string_sets and word in firms_string_sets[firm]:
                        count += 1
                if count == 2:
                    final_keyword[key].append(word)
        

        for key, value in final_keyword.items():
            print(key)
            print(value)
            print()
    


main()