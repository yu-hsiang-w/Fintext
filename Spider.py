import argparse
import csv
import json
import os
import pickle
import time
import random
from collections import defaultdict
from multiprocessing.pool import Pool

import spacy
import transformers
from transformers import AutoTokenizer

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

STOPWORDS = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'also', 'could', 'would'}


class DocumentProcessor:
    def __init__(self, tokenizer, compute_recurring_spans=True, min_span_length=1, max_span_length=10,
                 validate_spans=True, include_sub_clusters=False):
        self.nlp = spacy.load("en_core_web_sm") if compute_recurring_spans else None
        self.tokenizer = tokenizer
        self.compute_recurring_spans = compute_recurring_spans
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.validate_spans = validate_spans
        self.include_sub_clusters = include_sub_clusters

    def _find_all_spans_in_document(self, doc_psgs):
        output = defaultdict(list)
        for psg in doc_psgs:
            psg_idx, psg_tokens = psg
            tokens_txt = [t.text.lower() for t in psg_tokens]
            for i in range(len(psg_tokens)):
                for j in range(i + self.min_span_length, min(i + self.max_span_length, len(psg_tokens))):
                    length = j - i
                    if self.validate_spans:
                        if (length == 1 and DocumentProcessor.validate_length_one_span(psg_tokens[i])) or (
                                length > 1 and DocumentProcessor.validate_ngram(psg_tokens, i, length)):
                            span_str = " ".join(tokens_txt[i: j])
                            output[span_str].append((psg_idx, i, j))
        return output

    @staticmethod
    def validate_ngram(tokens, start_index, length):
        if any((not tokens[idx].is_alpha) and (not tokens[idx].is_digit) for idx in
               range(start_index, start_index + length)):
            return False

        # We filter out n-grams that are all stopwords, or that begin or end with stop words (e.g. "in the", "with my", ...)
        if any(tokens[idx].text.lower() not in STOPWORDS for idx in range(start_index, start_index + length)) and \
                tokens[start_index].text.lower() not in STOPWORDS and tokens[
            start_index + length - 1].text.lower() not in STOPWORDS:
            return True

        # TODO: Consider validating that the recurring span is not contained in the title (and vice versa)
        # span_lower = span.lower()
        # title_lower = title.lower()
        # if span_lower in title_lower or title_lower in span_lower:
        #     return False
        return False

    @staticmethod
    def validate_length_one_span(token):
        return token.text[0].isupper() or (len(token.text) == 4 and token.is_digit)

    @staticmethod
    def _filter_sub_clusters(recurring_spans):
        output = []
        span_txts = recurring_spans.keys()
        span_txts = sorted(span_txts, key=lambda x: len(x))
        for idx, span_txt in enumerate(span_txts):
            locations = recurring_spans[span_txt]
            is_sub_span = False
            for larger_span_txt in span_txts[idx + 1:]:
                if span_txt in larger_span_txt:
                    larger_locations = recurring_spans[larger_span_txt]
                    if len(locations) != len(larger_locations):
                        continue
                    is_different = False
                    for i in range(len(locations)):
                        psg_idx, start, end = locations[i]
                        larger_psg_idx, larger_start, larger_end = larger_locations[i]
                        if not (psg_idx == larger_psg_idx and start >= larger_start and end <= larger_end):
                            is_different = True
                            break
                    if not is_different:
                        is_sub_span = True
                        break
            if not is_sub_span:
                output.append((span_txt, locations))
        return output

    def _find_recurring_spans_in_documents(self, doc_psgs):
        """
        This function gets a list of spacy-tokenized passages and returns the list of recurring spans that appear in
        more than one passage
        Returns: A list of tuples, each representing a recurring span and has two items:
        * A string representing the lower-cased version of the recurring span
        * A list of it occurrences, each represented with a three-item tuple: (psg_index, span_start, span_end)
        """
        # first we get all spans with length >= min_length and validated (if wanted)
        spans_txts_to_locations = self._find_all_spans_in_document(doc_psgs)
        # now we filter out the spans that aren't recurring (or are recurring, but only in one passage)
        recurring = {}
        for span_txt, locations in spans_txts_to_locations.items():
            if len(locations) > 1:
                first_occurrence = locations[0][0]
                # check if span occurs in more than one passage
                for location in locations[1:]:
                    if location[0] != first_occurrence:
                        recurring[span_txt] = locations
                        break
        if self.include_sub_clusters:
            return recurring
        # else, filter out sub_clusters
        output = self._filter_sub_clusters(recurring)
        return output

    def _encode_and_convert_span_indices(self, spacy_tokenized_psgs, title, recurring_spans):
        encoded_psgs = {}
        old_to_new_indices = {}
        encoded_title = self.tokenizer.encode(title, add_special_tokens=False)
        for psg_id, psg in spacy_tokenized_psgs:
            encoded_psg = []
            indices_map = []
            for token in psg:
                new_idx = len(encoded_psg)
                indices_map.append(new_idx)
                encoded_psg.extend(self.tokenizer.encode(token.text, add_special_tokens=False))
            encoded_psgs[psg_id] = (encoded_title, encoded_psg)
            old_to_new_indices[psg_id] = indices_map

        new_recurring_spans = []
        for span_str, span_occurrences in recurring_spans:
            '''
            new_span_occurrences = [
                (psg_index, old_to_new_indices[psg_index][span_start], old_to_new_indices[psg_index][span_end])
                for psg_index, span_start, span_end in span_occurrences]
            '''
            new_span_occurrences = [
                (psg_index, old_to_new_indices[psg_index][span_start] if span_start < len(old_to_new_indices[psg_index]) else None, 
                old_to_new_indices[psg_index][span_end-1] if span_end-1 < len(old_to_new_indices[psg_index]) else None)
                for psg_index, span_start, span_end in span_occurrences
                if span_start < len(old_to_new_indices[psg_index]) and span_end <= len(old_to_new_indices[psg_index])
            ]
            new_recurring_spans.append((span_str, new_span_occurrences))

        return encoded_psgs, new_recurring_spans

    def _get_candidates_for_recurring_spans(self, psgs):
        """
        This function removes articles with less than a given number of passages.
        In addition, it removes the last passage because it also contains the prefix of the article (which is problematic
        for recurring span identification)
        """
        psgs = psgs[:-1]

        if len(psgs) < 3:
            return None
        return psgs

    def _postprocess_recurring_span_examples(self, recurring_spans, all_candidate_psgs, title):
        new_recurring_spans = []
        for span_str, span_occurrences in recurring_spans:
            positive_ctxs = set([psg_index for psg_index, _, __ in span_occurrences])
            if len(positive_ctxs) < 2:
                continue
            negative_ctxs = list(set([psg_index for psg_index, _ in all_candidate_psgs]) - positive_ctxs)
            if len(negative_ctxs) < 1:
                continue
            new_recurring_spans.append({
                "span": span_str,
                "title": title,
                "positive": span_occurrences,
                "negative_ctxs": negative_ctxs
            })
        return new_recurring_spans

    def process_document(self, psgs, title):
        """
        This function gets a list of string corresponding to passages.
        It tokenizes them and finds clusters of recurring spans across the passages
        """
        if self.compute_recurring_spans:
            tokenized_psgs = [(psg_id, self.nlp(psg_txt)) for psg_id, psg_txt in psgs]
            psgs_for_recurring_spans = self._get_candidates_for_recurring_spans(tokenized_psgs)
            if psgs_for_recurring_spans is not None:
                recurring_spans = self._find_recurring_spans_in_documents(psgs_for_recurring_spans)
            else:
                recurring_spans = []
            encoded_psgs, recurring_spans = self._encode_and_convert_span_indices(tokenized_psgs, title, recurring_spans)
            recurring_spans = self._postprocess_recurring_span_examples(recurring_spans, psgs_for_recurring_spans, title)
            return encoded_psgs, recurring_spans
        else:
            encoded_psgs = {}
            encoded_title = self.tokenizer.encode(title, add_special_tokens=False)
            for psg_id, psg_txt in psgs:
                encoded_psg_txt = self.tokenizer.encode(psg_txt, add_special_tokens=False)
                encoded_psgs[psg_id] = (encoded_title, encoded_psg_txt)
            return encoded_psgs, []


def preprocess_shard(shard_idx, args, article_to_psgs, tokenizer, compute_recurring_spans=True):
    all_recurring_spans = []
    
    processor = DocumentProcessor(tokenizer,
                                  compute_recurring_spans=compute_recurring_spans,
                                  max_span_length=args.max_span_length,
                                  min_span_length=args.min_span_length,
                                  validate_spans=True,
                                  include_sub_clusters=False)
    
    for i, (title, psgs) in enumerate(article_to_psgs.items()):
        encoded_psgs, recurring_spans = processor.process_document(psgs, title)

    for recurring_span in recurring_spans:
        all_recurring_spans.append(recurring_span['span'])

    return all_recurring_spans


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    existed_dataset = []
    existed_dataset1 = []
    existed_dataset2 = []
    directory1 = "/home/ythsiao/output"
    firm_list = os.listdir(directory1)

    for firm in firm_list:
        print(firm)
        directory2 = os.path.join(directory1, firm)
        directory3 = os.path.join(directory2, "10-K")
        tenK_list = os.listdir(directory3)
        tenK_list = sorted(tenK_list)


        for tenK in tenK_list:
            print(tenK)
            if random.random() < 0.01:
                if tenK[:4] != "2023":
                    file_path = os.path.join(directory3, tenK)
                    with open(file_path, 'r') as file:
                        article_to_psgs = {}
                        for line in file:
                            data = json.loads(line)
                            if 'company_name' in data:
                                tenKid = data['company_name'] + " " + data['filing_date']
                            if 'paragraph' in data:
                                concatenated_string = " ".join(data["paragraph"])
                                if tenKid not in article_to_psgs:
                                    article_to_psgs[tenKid] = [(1,concatenated_string)]
                                else:
                                    dict_length = len(article_to_psgs[tenKid])
                                    article_to_psgs[tenKid].append((dict_length,concatenated_string))

                    if len(article_to_psgs) != 0:
                        shard_idx = 0
                        all_recurring_spans = preprocess_shard(shard_idx, args, article_to_psgs, tokenizer, compute_recurring_spans=True)
                        
                        for recurring_span in all_recurring_spans:
                            count = 0
                            hard_neg = ""
                            with open(file_path, 'r') as file:
                                lines = file.readlines()
                            
                            random.shuffle(lines)

                            for line in lines:
                                data = json.loads(line)
                                if 'paragraph' in data:
                                    concatenated_string = " ".join(data["paragraph"])
                                    if (count == 0) and (recurring_span in concatenated_string):
                                        first_view = concatenated_string
                                        count += 1
                                    elif (count == 1) and (recurring_span in concatenated_string):
                                        second_view = concatenated_string
                                        count += 1
                                    elif (count < 2) and (recurring_span not in concatenated_string):
                                        hard_neg = concatenated_string
                            
                            existed_dataset1 = append_data(
                                existed_dataset=existed_dataset1,
                                question=first_view,
                                positive_ctxs_title="",
                                positive_ctxs_text=second_view,
                                negative_ctxs_title="",
                                negative_ctxs_text="",
                                hard_negative_ctxs_title="",
                                hard_negative_ctxs_text=hard_neg
                            )

    random.shuffle(existed_dataset1)

    existed_dataset1 = existed_dataset1[:7500]

    for firm in firm_list:
        print(firm)
        directory2 = os.path.join(directory1, firm)
        directory3 = os.path.join(directory2, "10-K")
        tenK_list = os.listdir(directory3)
        tenK_list = sorted(tenK_list)


        for tenK in tenK_list:
            print(tenK)
            if random.random() < 0.01:
                if tenK[:4] != "2023":
                    file_path = os.path.join(directory3, tenK)
                    with open(file_path, 'r') as file:
                        article_to_psgs = {}
                        for line in file:
                            data = json.loads(line)
                            if 'company_name' in data:
                                tenKid = data['company_name'] + " " + data['filing_date']
                            if 'paragraph' in data:
                                concatenated_string = " ".join(data["paragraph"])
                                if tenKid not in article_to_psgs:
                                    article_to_psgs[tenKid] = [(1,concatenated_string)]
                                else:
                                    dict_length = len(article_to_psgs[tenKid])
                                    article_to_psgs[tenKid].append((dict_length,concatenated_string))

                    if len(article_to_psgs) != 0:
                        shard_idx = 0
                        all_recurring_spans = preprocess_shard(shard_idx, args, article_to_psgs, tokenizer, compute_recurring_spans=True)
                        
                        for recurring_span in all_recurring_spans:
                            count = 0
                            hard_neg = ""
                            with open(file_path, 'r') as file:
                                lines = file.readlines()

                            random.shuffle(lines)
                                
                            for line in lines:
                                data = json.loads(line)
                                if 'paragraph' in data:
                                    concatenated_string = " ".join(data["paragraph"])
                                    if (count == 0) and (recurring_span in concatenated_string):
                                        first_view = concatenated_string.replace(recurring_span, "")
                                        count += 1
                                    elif (count == 1) and (recurring_span in concatenated_string):
                                        second_view = concatenated_string
                                        count += 1
                                    elif (count < 2) and (recurring_span not in concatenated_string):
                                        hard_neg = concatenated_string
                            
                            existed_dataset2 = append_data(
                                existed_dataset=existed_dataset2,
                                question=first_view,
                                positive_ctxs_title="",
                                positive_ctxs_text=second_view,
                                negative_ctxs_title="",
                                negative_ctxs_text="",
                                hard_negative_ctxs_title="",
                                hard_negative_ctxs_text=hard_neg
                            )

    random.shuffle(existed_dataset2)

    existed_dataset2 = existed_dataset2[:7500]

    existed_dataset = existed_dataset1 + existed_dataset2
    
    with open('Positive Pairs Spider2.json', 'w') as f:
        json.dump(existed_dataset, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--compute_recurring_spans", action="store_true")
    parser.add_argument("--min_span_length", type=int, default=3)
    parser.add_argument("--max_span_length", type=int, default=10)
    parser.add_argument("--num_processes", type=int, default=64)

    args = parser.parse_args()
    main(args)