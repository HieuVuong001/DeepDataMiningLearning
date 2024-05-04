import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser(description='simple inference finedtuned seq 2 seq')

parser.add_argument('--model_checkpoint', type=str, required=True,
                    help='Various model checkpoints can be found on Hugging Face i.e T5, BART, ...')
parser.add_argument('--model_cachedir', type=str, default="./cache",
                     help='Path to your cache storage')
parser.add_argument('--pretrain_path', type=str, required=True,
                     help='Path to your model pretrained weights (.pth files)')

args = parser.parse_args()


# Load model and tokenizer from Hugging Face
model_checkpoint = args.model_checkpoint
modelcache_dir = args.model_cachedir
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir=modelcache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=modelcache_dir)

# Load finetuned model from local machine
pretrain_path = args.pretrain_path
checkpoint = torch.load(pretrain_path, map_location='cpu')

# Load weights into model
model.load_state_dict(checkpoint['model_state_dict'])

p_to_summarize = """
Professors: We are going to have a popquiz.
Students: Will the QUIZ be on Thursday?
Professors: Yes!
"""

# Tokenize the input
input_tokens = tokenizer(p_to_summarize, max_length=384, padding='max_length', truncation=True)

# Add batch so the model can generate
input_ids = torch.tensor(input_tokens['input_ids']).unsqueeze(0)
attention_mask = torch.tensor(input_tokens['attention_mask']).unsqueeze(0)

# Generate output tokens
generated_tokens = model.generate(input_ids, attention_mask=attention_mask, max_length=384).flatten()

print(tokenizer.decode(token_ids=generated_tokens, skip_special_tokens=True))