import argparse
import os
import sys
import json
import evaluate

parse = argparse.ArgumentParser()

parse.add_argument('--metric', type=str, default='bleu', help='metric to evaluate')
parse.add_argument("--task", type=str, default="summarization", help="task of the model")
parse.add_argument("--model_type", type=str, choices=["tuned", "base"], help="type of the model")
parse.add_argument('--pred_file_path', type=str, default='pred.jsonl', help='prediction file')
parse.add_argument('--ref_file_path', type=str, default='ref.jsonl', help='reference file')
parse.add_argument('--output_path', type=str, default='output.txt', help='output file')
parse.add_argument("--src_column", type=str, default="src", help="column of source in the file")
parse.add_argument("--pred_column", type=str, default="pred", help="column of prediction in the file")
parse.add_argument("--ref_column", type=str, default="ref", help="column of reference in the file")
parse.add_argument("--special_columns", type=str, default="src", help="special columns in the file")

opt = parse.parse_args()

metric = evaluate.load(opt.metric.lower())

# load data
preds = []
with open(opt.pred_file_path, 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        preds.append(json_obj)
refs = []
with open(opt.ref_file_path, 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        refs.append(json_obj)

special_sentences, ref_sentences, pred_sentences = [], [], []
# type check
assert len(refs) == len(preds), "length not match"
for ref, pred in zip(refs, preds):
    assert pred[opt.src_column] == ref[opt.src_column], "input not match"
    assert opt.pred_column in pred, "prediction column not found"
    assert opt.ref_column in ref, "reference column not found"
    # special columns for sari
    if opt.metric.lower() == "sari":
        assert "info" in ref, "info not found"
        assert opt.special_columns in ref["info"], "special columns not found"
        special_sentences.append(ref["info"][opt.special_columns])
    ref_sentences.append(ref[opt.ref_column])
    pred_sentences.append(pred[opt.pred_column])

if opt.metric.lower() == "bleu" or opt.metric.lower() == "sari":
    if type(ref_sentences[0]) != list:
        # bleu and meteor need multiple references
        ref_sentences = [[ref] for ref in ref_sentences]
    assert type(ref_sentences[0][0]) == str, "multiple references needed"
# get evaluation result
if opt.metric.lower() == "sari":
    # sari need multiple references
    result = metric.compute(sources=special_sentences, predictions=pred_sentences, references=ref_sentences)
else:
    result = metric.compute(predictions=pred_sentences, references=ref_sentences)
# save result
save_content = {"task": opt.task, "model_type": opt.model_type, "metric": opt.metric, "result": result}

output_dir = os.path.dirname(opt.output_path)
os.makedirs(output_dir, exist_ok=True)
with open(opt.output_path, 'a') as f:
    f.write(json.dumps(save_content) + '\n')



