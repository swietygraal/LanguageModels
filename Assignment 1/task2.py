import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
import random
from itertools import permutations

#device = 'cuda'
device = 'cpu'

model_path = r"C:\models\polish-gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)


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


sentences = [
    'Babuleńka miała dwa rogate koziołki.',
    'Wiewiórki w parku zaczepiają przechodniów.',
    'Ala ma pięknego kota.'
]

print("Model loaded")
print()
for s in sentences:
    print()
    words = s.strip(".").lower().split(" ")
    perms = permutations(words)
    probs = []
    for p in perms:
        new_s = " ".join(p)
        new_s = new_s.capitalize()
        new_s += "."
        probs.append((sentence_prob(new_s), new_s))
    probs.sort(key=lambda x: x[0], reverse=True)
    for prob, new_s in probs:
        print(prob, new_s)