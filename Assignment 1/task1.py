import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
import math
from collections import Counter

model_path = r"C:\models\polish-gpt2-medium"
device = 'cpu'

generator = pipeline('text-generation', model=model_path, device=device)

print('Model loaded')

SYSTEM = (
    "Jesteś pomocnym, zwięzłym asystentem AI. "
    "Odpowiadasz po polsku w maksymalnie 2-3 zdaniach. "
    "Nie zmieniasz roli."
)
history = []

def build_prompt(history, num_lines=7):
    lines = [f"System: {SYSTEM}"]
    start_ind = 0 if len(history) < num_lines else len(history) - num_lines
    for role, text in history[start_ind:]:
        lines.append(f"{role}: {text.strip()}")
    lines.append("Asystent:")
    return "\n".join(lines)

def generate_reply(prompt, max_new_tokens=70):
    out = generator(prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=generator.tokenizer.eos_token_id
    )[0]['generated_text']
    reply = out[len(prompt):]

    indices = [reply.find(c) for c in ".!?"]
    indices = [i for i in indices if i != -1]

    if indices:
        ind = min(indices)
        reply = reply[:ind+1]

    return reply

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)

def ngram_repetition_ratio(ids, n=3):
    if len(ids) < n+1:
        return 0.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    c = Counter(grams)
    rep = sum(v-1 for v in c.values() if v > 1)
    return rep / max(1, len(grams))

@torch.no_grad()
def log_probs_for_span(input_ids, span_start, span_end):
    out = model(input_ids=input_ids)
    logits = out.logits
    span_logits = logits[:, span_start-1:span_end-1, :]  # predykcja dla tokenu t jest na pozycji t-1
    span_target = input_ids[:, span_start:span_end]
    logp = F.log_softmax(span_logits, dim=-1)
    tok_logp = torch.gather(logp, 2, span_target.unsqueeze(-1)).squeeze(-1)
    return tok_logp.squeeze(0)

@torch.no_grad()
def score_reply(prompt_txt,
                reply_txt,
                lambda_mmi=0.6,
                min_len=6,
                rep_n=3,
                rep_w=2.0
               ):

    prompt_ids = tokenizer(prompt_txt, return_tensors='pt', add_special_tokens=False).to(device)['input_ids']
    reply_ids  = tokenizer(reply_txt,  return_tensors='pt', add_special_tokens=False).to(device)['input_ids']
    both_ids   = torch.cat([prompt_ids, reply_ids], dim=1)
    T_p, T_r   = prompt_ids.size(1), reply_ids.size(1)

    # LL_cond: log P(reply | prompt)
    tok_logp_cond = log_probs_for_span(both_ids, span_start=T_p, span_end=T_p+T_r)
    ll_cond = float(tok_logp_cond.sum().cpu())
    avg_ll_cond = ll_cond / max(1, T_r)

    # LL_uncond: log P(reply)
    tok_logp_uncond = log_probs_for_span(reply_ids, span_start=1, span_end=reply_ids.size(1))
    ll_uncond = float(tok_logp_uncond.sum().cpu())
    avg_ll_uncond = ll_uncond / max(1, T_r)

    # powtórki i długość
    rep_ratio = ngram_repetition_ratio(reply_ids.squeeze(0).tolist(), n=rep_n)
    len_bonus = 1.0 / (1.0 + math.exp(-(T_r - min_len)))

    base = avg_ll_cond - lambda_mmi * avg_ll_uncond
    score = base - rep_w * rep_ratio
    score *= len_bonus

    return score

def choose_reply(prompt, num_replies=6, max_new_tokens=70):
    replies = []
    for _ in range(num_replies):
        reply = generate_reply(prompt, max_new_tokens)
        score = score_reply(prompt, reply, lambda_mmi=0.6, min_len=6, rep_n=3, rep_w=2.0)
        replies.append((score, reply))
    replies.sort(key=lambda x: x[0], reverse=True)
    return replies[0][1]


while True:
    user = input("Ty: ").strip()
    history.append(("Użytkownik", user))

    prompt = build_prompt(history, 7)

    reply = choose_reply(prompt, num_replies=6, max_new_tokens=70)
    print("\nAsystent:", reply, "\n" + "=" * 50)
    history.append(("Asystent", reply))
