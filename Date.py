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

def contains_single_date(string):
    # Check if the string contains "December 31" or "January 1"
    if re.search(r'\b(?:December 31|January 1), \d{4}\b', string, re.IGNORECASE):
        return "NA"

    # The regular expression pattern for the date format "February 8, 2022"
    pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'
    
    # Find all occurrences of the date pattern in the string
    dates = re.findall(pattern, string, re.IGNORECASE)

    # Check if there is exactly one date in the string
    if len(dates) == 1:
        return dates[0]
    else:
        return "NA"

def remove_date_pattern(text):
    pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'
    modified_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return modified_text


def append_data(existed_dataset, question, positive_ctxs_title, positive_ctxs_text, negative_ctxs_title, negative_ctxs_text, hard_negative_ctxs_title, hard_negative_ctxs_text):
    # Assuming the structure of the existing dataset to be similar to the provided data
    new_entry = {
        "question": question,
        "positive_ctxs": [{
            "title": positive_ctxs_title,
            "text": positive_ctxs_text
        }],
        "negative_ctxs": [{
            "title": negative_ctxs_title,
            "text": negative_ctxs_text,
        }],
        "hard_negative_ctxs": [{
            "title": hard_negative_ctxs_title,
            "text": hard_negative_ctxs_text,
        }]
    }

    # Append the new entry to the existed dataset
    existed_dataset.append(new_entry)

    return existed_dataset

def main():

    directory1 = "/home/ythsiao/output"
    firm_list = os.listdir(directory1)
    existed_dataset = []
    all_firms_special_dates = {}

    total_count = 0

    for firm in firm_list:
        print(firm)
        firm_name = ""
        special_dates = {}
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
                            firm_name  = data['company_name']
                        if 'paragraph' in data:
                                concatenated_string = " ".join(data["paragraph"])
                                #concatenated_string = firm_name + " " + concatenated_string
                                result = contains_single_date(concatenated_string)
                                number_ratio = (sum(1 for character in concatenated_string if character.isdigit())) / len(concatenated_string)
                                if (result != "NA") and (result in special_dates) and (number_ratio < 0.1):
                                    special_dates[result].append(concatenated_string)
                                elif (result != "NA") and (result not in special_dates) and (number_ratio < 0.1):
                                    special_dates[result] = [concatenated_string]

        for key, value in special_dates.items():
            special_dates[key] = list(set(value))

        if firm not in all_firms_special_dates:
            all_firms_special_dates[firm] = special_dates

    for firm, dic in all_firms_special_dates.items():
        for key, value in dic.items():
            if len(value) > 1 and len(value) < 6:
                if len(value) > 2:
                    for i in range(len(value)):
                        if i != (len(value) - 1):
                            first_view = value[i]
                            second_view = value[i+1]
                        elif i == (len(value) - 1):
                            first_view = value[i]
                            second_view = value[0]
                        '''
                        for firm2, dic2 in all_firms_special_dates.items():
                            if (firm != firm2) and (key in dic2):
                                hard_negative = random.choice(dic2[key])
                        '''

                        first_view = remove_date_pattern(first_view)
                        second_view = remove_date_pattern(second_view)

                        print(key)
                        print()
                        print(first_view)
                        print()
                        print(second_view)
                        print()
                        print()
                        print()
                                    

                        total_count = total_count + 1
                        
                        existed_dataset = append_data(
                            existed_dataset=existed_dataset,
                            question=first_view,
                            positive_ctxs_title="",
                            positive_ctxs_text=second_view,
                            negative_ctxs_title="",
                            negative_ctxs_text="",
                            hard_negative_ctxs_title="",
                            hard_negative_ctxs_text=""
                        )

                elif len(value) == 2:
                    first_view = value[0]
                    second_view = value[1]
                    
                    '''
                    for firm2, dic2 in all_firms_special_dates.items():
                        if (firm != firm2) and (key in dic2):
                            hard_negative = random.choice(dic2[key])
                    '''

                    first_view = remove_date_pattern(first_view)
                    second_view = remove_date_pattern(second_view)

                    print(key)
                    print()
                    print(first_view)
                    print()
                    print(second_view)
                    print()
                    print()
                    print()
                                

                    total_count = total_count + 1
                    
                    existed_dataset = append_data(
                        existed_dataset=existed_dataset,
                        question=first_view,
                        positive_ctxs_title="",
                        positive_ctxs_text=second_view,
                        negative_ctxs_title="",
                        negative_ctxs_text="",
                        hard_negative_ctxs_title="",
                        hard_negative_ctxs_text=""
                    )


                

    print(total_count)
    
    with open('contrastive embedding data.json', 'w') as f:
        json.dump(existed_dataset, f, indent=2)
    





main()