### 这个文件适用于查看Evaluator.py的输出 ___Hancy


import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from Evaluator import BiasEvaluator, ScoreEvaluator

model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
evaluator = BiasEvaluator(tokenizer=tokenizer)
results = evaluator.evaluate(model)
score_evaluator = ScoreEvaluator(results)
overall = score_evaluator.get_overall_results()

print(json.dumps(overall, indent=2))
pres = score_evaluator.pretty_print(overall)
print("pretty_print output:", pres)