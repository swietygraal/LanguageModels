import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
import random
from itertools import permutations

model_name = r"C:\models\polish-gpt2-medium"
#device = 'cuda'
device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label


def sentence_prob(sentence_txt):
    input_ids = tokenizer(sentence_txt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids)
        log_probs = log_probs_from_logits(output.logits[:, :-1, :], input_ids[:, 1:])
        seq_log_probs = torch.sum(log_probs)
    return seq_log_probs.cpu().numpy()


with open("reviews_for_task3.txt", "r", encoding="utf-8") as f:
    sentences = f.read().splitlines()

#sentences = [
#    'GOOD Lokalizacja w centrum - przy głównym deptaku, po sąsiedzku Biedronka : ) MINUSYDługie schody do pokonania ( od wejścia głównego hotelu do recepcji ).',
#    'GOOD Hotel jest położony prawie nad samym jeziorem, u jego początku.',
#    'GOOD Warto przejechać ponad 600 km żeby przez tydzień wypocząć w tym hotelu : )'
#]

print()
correct = 0
for s in sentences:
    first_space = s.find(" ")
    sentiment = s[:first_space]
    review = s[first_space+1:]
    good_prob = sentence_prob(review + " Polecam!")
    bad_prob = sentence_prob(review + " Nie polecam!")
    if (good_prob > bad_prob and sentiment == "GOOD") or (bad_prob > good_prob and sentiment == "BAD"):
        correct += 1

print(correct, correct/len(sentences) * 100)
