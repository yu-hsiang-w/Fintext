import os
import torch
import torch.nn as nn
import transformers
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader, RandomSampler
import random
import json
import normalize_text


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


class InBatch(nn.Module):
    def __init__(self, retriever, tokenizer):
        super(InBatch, self).__init__()

        self.tokenizer = tokenizer
        self.encoder = retriever

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask)

        scores = torch.einsum("id, jd->ij", qemb / 0.05, kemb)
        print(scores[-1])

        loss = torch.nn.functional.cross_entropy(scores, labels)
        
        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        print(accuracy)

        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        print(stdq)
        print(stdk)

        return loss


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


def set_optim(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)

    scheduler_args = {
        "warmup": 1170,
        "total": 23400,
        "ratio": 0.0,
    }
    scheduler_class = WarmupLinearScheduler
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        training=False,
        maxload=None,
        normalize=False,
    ):
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.normalize_fn = normalize_text.normalize
        self.training = training
        self._load_data(datapaths, maxload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        gold = example["positive_ctxs"][0]
        negatives = [example["hard_negative_ctxs"][0]]

        gold = gold["title"] + " " + gold["text"] if "title" in gold and len(gold["title"]) > 0 else gold["text"]

        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]

        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives],
        }
        return example

    def _load_data(self, datapaths, maxload):
        counter = 0
        self.data = []
        files = os.listdir(datapaths)
        for path in files:
            path = str(path)
            file_data, counter = self._load_data_json(path, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, counter, maxload=None):
        examples = []

        current_directory = os.getcwd()
        directory = os.path.join(current_directory, "Formatted Data")
        path = os.path.join(directory, path)
        #path = "/home/yhwang/FinText/Formatted Data/contrastive embedding data.json"

        print("Before Load")
        with open(path, "r") as fin:
            data = json.load(fin)
        print("After Load")
        for example in data:
            counter += 1
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter


class Collator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            #"g_tokens": g_tokens,
            #"g_mask": g_mask,
            #"n_tokens": n_tokens,
            #"n_mask": n_mask,
        }

        return batch

def finetuning(model, optimizer, scheduler, tokenizer, step):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("11")
    current_directory = os.getcwd()
    train_data = os.path.join(current_directory, "Formatted Data")
    train_dataset = Dataset(
        datapaths=train_data,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        normalize=False,
        maxload=None,
        training=True
    )
    print("12")
    collator = Collator(tokenizer, passage_maxlength=256)
    print("13")
    train_sampler = RandomSampler(train_dataset)
    print("14")
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=32,
        drop_last=True,
        num_workers=3,
        collate_fn=collator
    )
    print("Before Training")
    model.train()
    epoch = 0
    while epoch < 50:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            print(f"{i}, {epoch}")
            #print("Training 1")
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            train_loss = model(**batch, stats_prefix="train")
            train_loss.backward()
            #print("Training 2")
            optimizer.step()
            scheduler.step()

            # Print the learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr}")

            optimizer.zero_grad()


def main():

    torch.manual_seed(2024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    step = 0
    print("Phase 1")
    retriever = Contriever.from_pretrained("bert-base-uncased")
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    model = InBatch(retriever, tokenizer)
    print("Phase 2")
    model = model.to(device)
    print("Phase 3")
    optimizer, scheduler = set_optim(model)
    print("Phase 4")
    finetuning(model, optimizer, scheduler, tokenizer, step)
    model_save_path = "/home/yhwang/FinText/Trained_Model"
    model.encoder.save_pretrained(model_save_path)
    print("Finish")


main()