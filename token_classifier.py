import torch
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer, AutoModel, BertForTokenClassification
import random
import os
import json
import math
import argparse
import faiss
import numpy as np
from scipy.stats import pearsonr
import pickle
import logging
import warnings

# Seed for reproducibility
random.seed(2024)

# Suppress specific warnings using the warnings module
warnings.filterwarnings('ignore', message="Some weights of .* were not initialized from the model checkpoint")

# Set up logging to suppress all warnings and only show errors
logging.getLogger("transformers").setLevel(logging.ERROR)


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

def embed_single_text(text, model, tokenizer, device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            text,
            max_length=256,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        
        inputs_tokens, inputs_mask = inputs["input_ids"], inputs["attention_mask"].bool()
        outputs = model(input_ids=inputs_tokens, attention_mask=inputs_mask)
        
        # Extract the embedding from the outputs
        if hasattr(outputs, 'last_hidden_state'):
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  
        else:
            emb = outputs[0][:, 0, :].cpu().numpy()  
        
    return emb

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, k):
    distances, indices = index.search(query_embedding, k)
    return indices, distances

def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

def main():

    # Argument parser setup
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_size', type=int, help='Size of training data', default=800)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1)

    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load annotated results
    annotated_results = []
    with open('fin.rag/annotation/annotated_result/all/aggregate_qlabels.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            id_text_label = [
                data['id'],
                data['text'],
                data['tokens'],
                data['highlight_labels'],
                data['highlight_probs']
            ]
            annotated_results.append(id_text_label)

    # Split data into training and testing sets
    training_elements = random.sample(annotated_results, args.training_size)
    testing_elements = [element for element in annotated_results if element not in training_elements]

    # Initialize tokenizer and retriever model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Initialize classification model
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device).train()

    # Setup optimizer and loss criterion
    optimizer = torch.optim.Adam(
        list(model.parameters()), lr=args.lr
    )
    criterion = torch.nn.BCELoss()
    
    epoch = args.epochs
    
    for j in range(epoch):
        count = 0
        for training_element in training_elements:            
            count += 1
            print(f'{j + 1}, {count}')
            
            # Tokenize training element
            tokenized_ids = tokenizer.convert_tokens_to_ids(training_element[2])
            tokenized_stringA = torch.tensor(tokenized_ids).unsqueeze(0).to(device)
            
            # Truncate tokenized strings if necessary
            if tokenized_stringA.size(1) > 250:
                tokenized_stringA = tokenized_stringA[:, :250]
            
            # Combine tokenized strings with separator token
            sep_token_id = tokenizer.sep_token_id
            sep_token_tensor = torch.tensor([[sep_token_id]]).to(device)
            combined_tokenized_string = torch.cat((tokenized_stringA, sep_token_tensor), dim=1).to(device)
            
            # Create attention mask
            attention_mask = torch.ones(combined_tokenized_string.shape, dtype=torch.long).to(device)
            
            inputs = {
                "input_ids": combined_tokenized_string,
                "attention_mask": attention_mask
            }
            
            # Get model outputs
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Find the separator token index
            sep_index = (combined_tokenized_string == sep_token_id).nonzero(as_tuple=True)[1].item()
            logits_before_sep = logits[:, :sep_index, :]
            
            # Compute softmax probabilities
            probabilities = F.softmax(logits_before_sep, dim=-1)
            probabilities_label_1 = probabilities[..., 1]
            
            # Get the true label tensor and truncate if necessary
            true_label_tensor = torch.tensor(training_element[3]).float().to(device)
            if len(true_label_tensor) > 250:
                true_label_tensor = true_label_tensor[:250]
            
            # Prepare for loss computation
            probabilities_label_1 = probabilities_label_1.view(-1).float()
            optimizer.zero_grad()
            
            # Compute and print loss
            loss = criterion(probabilities_label_1, true_label_tensor)
            print(loss)
            
            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
            
            


    num_of_good = 0
    total_correlation = 0
    total_precision = 0

    for testing_element in testing_elements:
        tokenized_ids = tokenizer.convert_tokens_to_ids(testing_element[2])
        tokenized_stringA = torch.tensor(tokenized_ids).unsqueeze(0).to(device)

        # Truncate sequences if necessary
        tokenized_stringA = tokenized_stringA[:, :250] if tokenized_stringA.size(1) > 250 else tokenized_stringA
        
        # Prepare combined input for the model
        sep_token_id = tokenizer.sep_token_id
        sep_token_tensor = torch.tensor([[sep_token_id]]).to(device)
        combined_tokenized_string = torch.cat((tokenized_stringA, sep_token_tensor), dim=1).to(device)
        attention_mask = torch.ones(combined_tokenized_string.shape, dtype=torch.long).to(device)
        
        # Get model outputs
        inputs = {"input_ids": combined_tokenized_string, "attention_mask": attention_mask}
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Calculate probabilities before the separator token
        sep_index = (combined_tokenized_string == sep_token_id).nonzero(as_tuple=True)[1].item()
        logits_before_sep = logits[:, :sep_index, :]
        probabilities = F.softmax(logits_before_sep, dim=-1)
        probabilities_label_1 = probabilities[..., 1]

        # Calculate correlation with true probabilities
        true_prob_tensor = torch.tensor(testing_element[4]).to(device)[:250]
        true_label_tensor = torch.tensor(testing_element[3]).to(device)[:250]
        
        array1 = probabilities_label_1.cpu().detach().numpy().flatten()
        array2 = true_prob_tensor.cpu().detach().numpy().flatten()
        correlation, _ = pearsonr(array1, array2)
        print(correlation)
        
        if not math.isnan(correlation):
            total_correlation += correlation
            num_of_good += 1

        # Calculate precision
        topk_indices = torch.topk(probabilities_label_1, torch.sum(true_label_tensor)).indices
        mask = torch.zeros_like(probabilities_label_1, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        binary_tensor = mask.float()
        predicted_label_tensor = binary_tensor.flatten()
        
        TP = ((true_label_tensor == 1) & (predicted_label_tensor == 1)).sum().item()
        FP = ((true_label_tensor == 0) & (predicted_label_tensor == 1)).sum().item()
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1
        print(precision)
        print()

        total_precision += precision

    # Print final average metrics
    print("Final Average Correlation:")
    print(total_correlation / num_of_good)
    print("Final Average Precision:")
    print(total_precision / len(testing_elements))



if __name__ == "__main__":
    main()