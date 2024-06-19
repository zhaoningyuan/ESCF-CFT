from huggingface_hub import InferenceClient
import argparse
import os, sys
import json
from functools import partial
import concurrent.futures
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='input.jsonl')
parser.add_argument('--output_file', type=str, default='output.jsonl')
parser.add_argument("--ip", type=str, default="http://127.0.0.1")
parser.add_argument("--port", type=int, default=8181)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_new_tokens", type=int, default=512)

args = parser.parse_args()

def gen_text(client, src):
    return client.text_generation(src, max_new_tokens=args.max_new_tokens, )


client = InferenceClient(model=":".join([args.ip, str(args.port)]))
gen_text_with_client = partial(gen_text, client)

srcs = []
with open(args.input_file, 'r') as f:
    for line in f:
        input_data = json.loads(line)
        srcs.append(input_data['prompt'])
preds = []
bs = args.batch_size
for i in tqdm(range(0, len(srcs), bs)):
    batch_src = srcs[i:i+bs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as executor: 
        out = list(executor.map(gen_text_with_client,batch_src))
    preds.extend(out)
# if args.output_file not exists, create it
output_dir = os.path.dirname(args.output_file)
os.makedirs(output_dir, exist_ok=True)
with open(args.output_file, 'w') as f:
    for src, pred in zip(srcs, preds):
        f.write(json.dumps({'prompt': src, 'pred': pred}) + '\n')

