import random
import os
import json

def random_cut_with_overlap(s):
    words = s.split()
    
    first_cut_index = random.randint(1, len(words))
    second_cut_index = random.randint(0, min(first_cut_index, len(words) - 1))
    
    first_half = ' '.join(words[:first_cut_index])
    second_half = ' '.join(words[second_cut_index:])
    
    return first_half, second_half

def tokenize(text):
    # This is a simple tokenizer; consider using a more sophisticated one for complex texts.
    return text.split()

def randomly_crop_span(s):
    if not s:
        return ""
    words = s.split()
    if len(words) == 1:
        return words[0]
    start_word_index = random.randint(0, len(words) - 2)
    end_word_index = random.randint(start_word_index + 1, len(words))
    return " ".join(words[start_word_index:end_word_index])

def sample_span(tokens):
    span_length = random.randint(1, len(tokens) - 2)
    start_index = random.randint(0, len(tokens) - span_length)
    return start_index, span_length

def inverse_cloze_task(tokens):
    start_index, span_length = sample_span(tokens)
    first_view = tokens[start_index:start_index + span_length]
    second_view = tokens[:start_index] + tokens[start_index + span_length:]
    return " ".join(first_view), " ".join(second_view)

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



existed_dataset = []
count = 0

directory1 = "/home/ythsiao/output"
firm_list = os.listdir(directory1)

#for i in range(min(15, len(firm_list))):
for firm in firm_list:
  #firm = firm_list[i]
  directory2 = os.path.join(directory1, firm)
  directory3 = os.path.join(directory2, "10-K")
  ten_k_list = os.listdir(directory3)
  for ten_k in ten_k_list:
    file_path = os.path.join(directory3, ten_k)
    with open(file_path, 'r') as file:
      for line in file:
        data = json.loads(line)
        if 'paragraph' in data:
          title = data["id"]
          text = data["paragraph"][0]
          tokens = tokenize(text)
          if len(tokens) > 3 and random.random() < 0.0045:
            count = count+1
            print(count)
            
            first_view, second_view = inverse_cloze_task(tokens)

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
            
            
            first_view = randomly_crop_span(text)
            second_view = randomly_crop_span(text)

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
            '''

            first_view, second_view = random_cut_with_overlap(text)

            existed_dataset = append_data(
                existed_dataset=existed_dataset,
                question=second_view,
                positive_ctxs_title="",
                positive_ctxs_text=first_view,
                negative_ctxs_title="",
                negative_ctxs_text="",
                hard_negative_ctxs_title="",
                hard_negative_ctxs_text=""
            )
            '''
print(len(existed_dataset))






with open('contrastive embedding data Contriever.json', 'w') as f:
    json.dump(existed_dataset, f, indent=2)
