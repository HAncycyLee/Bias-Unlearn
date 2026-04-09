from transformers import AutoTokenizer
from dataloader_mqa import (
    MQAJsonlDataset,
    MQASequenceClassificationCollator,
    create_mqa_dataloader,
)

tkmodel = '' # to be filled
mqa_jsonl_path = '' # to be filled

tokenizer = AutoTokenizer.from_pretrained(tkmodel)

# 下面是常用示例代码

dataset = MQAJsonlDataset(
    mqa_jsonl_path,
    task_types=["rule_choice"],
    difficulties=["easy", "medium", "hard"],
)

collator = MQAMultipleChoiceCollator(
    tokenizer=tokenizer,
    max_length=1024,
)

dataloader = create_mqa_dataloader(
    dataset=dataset,
    collator=collator,
    batch_size=4,
    shuffle=True,
)

# 下面是多选示例代码

dataset = MQAJsonlDataset(
    "data/mqa/mqa_rule_choice.jsonl",
    task_types=["rule_choice"],
    difficulties=["easy", "medium", "hard"],
)

collator = MQASequenceClassificationCollator(
    tokenizer=tokenizer,
    max_length=1024,
)

dataloader = create_mqa_dataloader(
    dataset=dataset,
    collate_fn=collator,
    batch_size=4,
    shuffle=True,
)