#dataloader_mqa.py
# encoding:utf-8
#   Data   : 2026-3-30

from __future__ import annotations

"""Dataloader utilities for MQA JSONL files.

This module is designed for the intermediate artifacts produced by
`build_mqa.py`, especially:
- unified_intermediate.jsonl
- mqa_rule_choice.jsonl
- mqa_score_band.jsonl

Design goals:
1. Keep the data layer independent from the current training head.
2. Support both sequence-classification style input and multiple-choice style input.
3. Preserve useful supervision beyond the hard label, such as score / difficulty / task type.

Typical usage:

    from transformers import AutoTokenizer
    from dataloader_mqa import (
        MQAJsonlDataset,
        MQASequenceClassificationCollator,
        MQAMultipleChoiceCollator,
        create_mqa_dataloader,
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    ds = MQAJsonlDataset(
        "data/mqa/mqa_rule_choice.jsonl",
        task_types=["rule_choice"],
        difficulties=["easy", "medium", "hard"],
    )

    collator = MQASequenceClassificationCollator(tokenizer, max_length=1024)
    dl = create_mqa_dataloader(ds, collator, batch_size=4, shuffle=True)

Returned batch keys for sequence classification:
- input_ids
- attention_mask
- labels                 (zero-based class ids)
- score_targets          (float tensor; NaN if unavailable)
- task_type_ids          (rule_choice=0, score_band=1)
- difficulty_ids         (easy=0, medium=1, hard=2)
- example_ids            (python list)
- label_letters          (python list)
- label_texts            (python list)
- raw_examples           (python list)

Returned batch keys for multiple choice:
- input_ids              (B, C, L)
- attention_mask         (B, C, L)
- labels                 (B,)
- score_targets          (B,)
- task_type_ids          (B,)
- difficulty_ids         (B,)
- num_choices            (B,)
- example_ids            (python list)
- raw_examples           (python list)
"""

import json
import math
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

SEED = 111

torch.manual_seed(SEED)  # 设置PyTorch随机种子
np.random.seed(SEED)     # 设置NumPy随机种子
random.seed(SEED)        # 设置Python内置random模块种子

SCHEMA_VERSION = "mqa-v1"
TASK_TYPE_TO_ID = {"rule_choice": 0, "score_band": 1}  # 定义任务类型到ID的映射
DIFFICULTY_TO_ID = {"easy": 0, "medium": 1, "hard": 2}  # 定义难度等级到ID的映射
TASK_LABEL_MAP = {
    "rule_choice": {"A": 0, "B": 1, "C": 2, "D": 3},      # 规则选择任务的标签映射
    "score_band": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},  # 分数区间任务的标签映射
}


@dataclass
class MQAExample:
    """MQA示例的数据类
    
    存储MQA数据集中的单个示例，包含任务相关信息如提示、选项、答案等。
    """

    schema_version: str
    task_type: str
    mqa_id: str
    source_item_id: str
    difficulty: str
    score: Optional[float]
    prompt: str
    choices: List[Dict[str, str]]
    answer: str
    metadata: Dict[str, Any]
    rule_alignment: Optional[str] = None
    score_band: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """将MQAExample实例转换为字典格式
        
        Returns:
            Dict[str, Any]: 包含MQAExample所有字段的字典
        """
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "mqa_id": self.mqa_id,
            "source_item_id": self.source_item_id,
            "difficulty": self.difficulty,
            "score": self.score,
            "prompt": self.prompt,
            "choices": self.choices,
            "answer": self.answer,
            "metadata": self.metadata,
            "rule_alignment": self.rule_alignment,
            "score_band": self.score_band,
        }


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """从JSONL文件中加载数据行
    
    Args:
        path: JSONL文件的路径
        
    Returns:
        List[Dict[str, Any]]: 加载的JSON对象列表
        
    Raises:
        ValueError: 当JSON解析失败时
    """

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):  # 从第1行开始计数
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))  # 解析JSON行
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _ensure_list(value: Optional[Sequence[str]]) -> Optional[List[str]]:
    """确保传入的值是列表格式
    
    Args:
        value: 可能为空或序列类型的输入值
        
    Returns:
        Optional[List[str]]: 转换后的列表或None
    """
    
    if value is None:
        return None
    return list(value)


def _render_choices(choices: Sequence[Dict[str, str]], with_ids: bool = True) -> str:
    """将选项渲染为字符串格式
    
    Args:
        choices: 包含选项ID和文本的字典序列
        with_ids: 是否在输出中包含选项ID
        
    Returns:
        str: 渲染后的选项字符串
    """

    lines: List[str] = []
    for choice in choices:
        cid = str(choice.get("id", "")).strip()
        text = str(choice.get("text", "")).strip()
        if with_ids and cid:
            lines.append(f"{cid}. {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def build_sequence_text(
    example: Dict[str, Any], 
    include_task_instruction: bool = True
    ) -> str:
    """构建用于序列分类的文本输入
    
    Args:
        example: 包含任务信息的字典
        include_task_instruction: 是否包含任务指令
        
    Returns:
        str: 构建好的文本输入
    """

    task_type = example["task_type"]
    header = []
    # 这里是构建MQA任务类型的地方，后面根据实验要求进一步优化
    if include_task_instruction:
        if task_type == "rule_choice":
            header.append("任务：根据题干与选项，判断回答更接近哪种规则立场。")
        elif task_type == "score_band":
            header.append("任务：根据题干与选项，判断回答的规则立场分数区间。")
        else:
            header.append(f"任务：{task_type}")

    header.append(f"题目：\n{example['prompt']}")  # 添加题目内容
    header.append("选项：")  # 添加选项标题
    header.append(_render_choices(example["choices"], with_ids=True))  # 渲染选项
    header.append("请输出最合适的选项字母。")  # 添加提示
    return "\n\n".join(header)  # 用双换行连接各部分


def build_multiple_choice_texts(
    example: Dict[str, Any], 
    include_task_instruction: bool = True
    ) -> List[str]:
    """构建用于多项选择的文本输入
    
    Args:
        example: 包含任务信息的字典
        include_task_instruction: 是否包含任务指令
        
    Returns:
        List[str]: 每个选项对应的文本列表
    """

    prompt_parts: List[str] = []
    if include_task_instruction:
        if example["task_type"] == "rule_choice":
            prompt_parts.append("任务：根据题干与候选选项，选择更合理的规则立场。")
        elif example["task_type"] == "score_band":
            prompt_parts.append("任务：根据题干与候选选项，选择最合理的分数区间。")
    prompt_parts.append(example["prompt"])  # 添加题目内容
    prompt_prefix = "\n\n".join(prompt_parts)  # 组合前缀

    rendered: List[str] = []
    for choice in example["choices"]:
        cid = str(choice.get("id", "")).strip()
        text = str(choice.get("text", "")).strip()
        rendered.append(f"{prompt_prefix}\n\n候选答案：{cid}. {text}")  # 每个选项构成一个完整输入
    return rendered


class MQAJsonlDataset(Dataset):
    """MQA JSONL文件的数据集类

    此数据集不自行分词，而是返回包含单个组合文本字段的Python字典，
    以及用于多项选择编码器的choice_texts。
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        task_types: Optional[Sequence[str]] = None,
        difficulties: Optional[Sequence[str]] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 111,
        require_score: bool = False,
    ) -> None:
        """初始化MQAJsonlDataset实例
        
        Args:
            jsonl_path: JSONL文件路径
            task_types: 限制的任务类型列表
            difficulties: 限制的难度级别列表
            max_samples: 最大样本数
            shuffle: 是否打乱数据
            seed: 随机种子，此处和前面的随机种子一致（保障复现）
            require_score: 是否要求必须有分数
        """

        self.jsonl_path = str(jsonl_path)
        self.task_types = _ensure_list(task_types)
        self.difficulties = _ensure_list(difficulties)
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        self.require_score = require_score

        rows = load_jsonl(self.jsonl_path)
        self.examples = self._prepare_examples(rows)

    def _prepare_examples(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备数据集示例
        
        过滤并处理原始数据行，生成符合要求的示例列表。
        
        Args:
            rows: 原始数据行列表
            
        Returns:
            List[Dict[str, Any]]: 处理后的示例列表
        """

        examples: List[Dict[str, Any]] = []
        for row in rows:
            task_type = row.get("task_type")
            if task_type not in TASK_LABEL_MAP:
                continue  # 跳过不在标签映射中的任务类型
            if self.task_types is not None and task_type not in self.task_types:
                continue  # 跳过不在指定任务类型中的行

            difficulty = row.get("difficulty", "medium") # 获取难度级别，默认为中等难度
            if self.difficulties is not None and difficulty not in self.difficulties:
                continue  # 跳过不在指定难度中的行

            score = row.get("score")
            if self.require_score and score is None:
                continue  # 跳过没有分数的行（如果要求必须有分数）

            answer_letter = row.get("answer")
            label_map = TASK_LABEL_MAP[task_type]
            if answer_letter not in label_map:
                continue  # 跳过答案不在标签映射中的行

            choices = row.get("choices", [])
            if not isinstance(choices, list) or not choices:
                continue  # 跳过没有有效选项的行

            processed = {
                "schema_version": row.get("schema_version", SCHEMA_VERSION),
                "task_type": task_type,
                "mqa_id": row.get("mqa_id", ""),
                "source_item_id": row.get("source_item_id", ""),
                "difficulty": difficulty,
                "score": float(score) if score is not None else None,
                "rule_alignment": row.get("rule_alignment"),
                "score_band": row.get("score_band"),
                "prompt": row.get("prompt", ""),
                "choices": choices,
                "answer": answer_letter,
                "label": label_map[answer_letter],  # 将答案字母转换为数字索引
                "label_text": self._lookup_label_text(choices, answer_letter),  # 获取答案文本
                "metadata": row.get("metadata", {}),
                "num_choices": len(choices),  # 计算选项数量
            }

            processed["text"] = build_sequence_text(processed)  # 构建序列文本
            processed["choice_texts"] = build_multiple_choice_texts(processed)  # 构建多项选择文本
            examples.append(processed)

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(examples)  # 打乱数据
        if self.max_samples is not None:
            examples = examples[: self.max_samples]  # 截取最大样本数
        return examples

    @staticmethod # 静态方法
    def _lookup_label_text(choices: Sequence[Dict[str, str]], answer_letter: str) -> str:
        """查找答案标签对应的文本内容
        
        Args:
            choices: 选项列表
            answer_letter: 答案字母
            
        Returns:
            str: 答案对应的文本内容
        """

        for choice in choices:
            if str(choice.get("id", "")).strip() == answer_letter:
                return str(choice.get("text", "")).strip()
        return ""

    def __len__(self) -> int:
        """获取数据集中示例的数量
        
        Returns:
            int: 数据集中示例的数量
        """

        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取指定索引处的示例
        
        Args:
            idx: 示例的索引
            
        Returns:
            Dict[str, Any]: 指定索引处的示例数据
        """

        return self.examples[idx]

    def get_label_map(self, task_type: str) -> Dict[str, int]:
        """获取指定任务类型的标签映射
        
        Args:
            task_type: 任务类型名称
            
        Returns:
            Dict[str, int]: 标签到索引的映射字典
        """
        
        return dict(TASK_LABEL_MAP[task_type])

    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息
        
        Returns:
            Dict[str, Any]: 包含数据集各种统计信息的字典
        """

        by_task: Dict[str, int] = {}
        by_difficulty: Dict[str, int] = {}
        for ex in self.examples:
            by_task[ex["task_type"]] = by_task.get(ex["task_type"], 0) + 1  # 统计任务类型分布
            by_difficulty[ex["difficulty"]] = by_difficulty.get(ex["difficulty"], 0) + 1  # 统计难度分布
        return {
            "jsonl_path": self.jsonl_path,
            "num_examples": len(self.examples),
            "task_distribution": by_task,
            "difficulty_distribution": by_difficulty,
        }


class MQASequenceClassificationCollator:
    """序列分类模型的数据整理器

    每个示例变成一个输入序列，包含提示和所有选项。
    黄金标签是正确选项字母的零基索引。
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> None:
        """初始化MQASequenceClassificationCollator实例
        
        Args:
            tokenizer: 分词器
            max_length: 最大长度
            padding: 是否填充
            truncation: 是否截断
            add_special_tokens: 是否添加特殊标记
        """

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将一批数据整理成模型可接受的格式
        
        Args:
            batch: 批次数据列表
            
        Returns:
            Dict[str, Any]: 整理后的批次数据
        """

        texts = [item["text"] for item in batch]  # 提取批次中的文本
        # 使用分词器对文本进行编码
        encoded = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",  # 返回PyTorch张量
            add_special_tokens=self.add_special_tokens,
        )

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)  # 转换标签为张量
        score_targets = torch.tensor(
            [float(item["score"]) if item["score"] is not None else float("nan") for item in batch],
            dtype=torch.float,  # 转换分数目标为张量
        )
        task_type_ids = torch.tensor(
            [TASK_TYPE_TO_ID[item["task_type"]] for item in batch], dtype=torch.long  # 转换任务类型为张量
        )
        difficulty_ids = torch.tensor(
            [DIFFICULTY_TO_ID.get(item["difficulty"], 1) for item in batch], dtype=torch.long  # 转换难度为张量
        )

        encoded["labels"] = labels
        encoded["score_targets"] = score_targets
        encoded["task_type_ids"] = task_type_ids
        encoded["difficulty_ids"] = difficulty_ids
        encoded["example_ids"] = [item["mqa_id"] for item in batch]  # 添加示例ID列表
        encoded["source_item_ids"] = [item["source_item_id"] for item in batch]  # 添加源项目ID列表
        encoded["label_letters"] = [item["answer"] for item in batch]  # 添加答案字母列表
        encoded["label_texts"] = [item["label_text"] for item in batch]  # 添加标签文本列表
        encoded["raw_examples"] = batch  # 保留原始示例
        return encoded


class MQAMultipleChoiceCollator:
    """标准多项选择模型的数据整理器

    输出形状：
    - input_ids: (batch, num_choices, seq_len)
    - attention_mask: (batch, num_choices, seq_len)
    - labels: (batch,)
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> None:
        """初始化MQAMultipleChoiceCollator实例
        
        Args:
            tokenizer: 分词器
            max_length: 最大长度
            padding: 是否填充
            truncation: 是否截断
            add_special_tokens: 是否添加特殊标记
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将一批数据整理成模型可接受的格式
        
        Args:
            batch: 批次数据列表
            
        Returns:
            Dict[str, Any]: 整理后的批次数据
        """

        num_choices = max(item["num_choices"] for item in batch)  # 获取批次中最大的选项数
        if len({item["num_choices"] for item in batch}) != 1:
            raise ValueError(
                "All examples in one batch must have the same number of choices for multiple-choice collation. "
                "Group rule_choice and score_band examples separately, or use the sequence collator instead."
            )

        flat_texts: List[str] = []  # 平铺所有文本
        for item in batch:
            flat_texts.extend(item["choice_texts"])  # 将每个示例的所有选项文本添加到列表

        encoded = self.tokenizer(
            flat_texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",  # 返回PyTorch张量
            add_special_tokens=self.add_special_tokens,
        )

        batch_size = len(batch)
        for key in list(encoded.keys()):  # 重塑张量形状为(batch_size, num_choices, seq_len)
            encoded[key] = encoded[key].view(batch_size, num_choices, -1)

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)  # 转换标签为张量
        score_targets = torch.tensor(
            [float(item["score"]) if item["score"] is not None else float("nan") for item in batch],
            dtype=torch.float,  # 转换分数目标为张量
        )
        task_type_ids = torch.tensor(
            [TASK_TYPE_TO_ID[item["task_type"]] for item in batch], dtype=torch.long  # 转换任务类型为张量
        )
        difficulty_ids = torch.tensor(
            [DIFFICULTY_TO_ID.get(item["difficulty"], 1) for item in batch], dtype=torch.long  # 转换难度为张量
        )

        encoded["labels"] = labels
        encoded["score_targets"] = score_targets
        encoded["task_type_ids"] = task_type_ids
        encoded["difficulty_ids"] = difficulty_ids
        encoded["num_choices"] = torch.tensor([item["num_choices"] for item in batch], dtype=torch.long)  # 选项数量张量
        encoded["example_ids"] = [item["mqa_id"] for item in batch]  # 添加示例ID列表
        encoded["source_item_ids"] = [item["source_item_id"] for item in batch]  # 添加源项目ID列表
        encoded["label_letters"] = [item["answer"] for item in batch]  # 添加答案字母列表
        encoded["raw_examples"] = batch  # 保留原始示例
        return encoded


def create_mqa_dataloader(
    dataset: Dataset,
    collate_fn,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """创建MQA数据加载器
    
    Args:
        dataset: 数据集
        collate_fn: 数据整理函数
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        drop_last: 是否丢弃最后一个不完整的批次
        
    Returns:
        DataLoader: 创建的数据加载器
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,  # 使用自定义的整理函数
    )


def build_dataset_from_paths(
    jsonl_paths: Union[str, Path, Sequence[Union[str, Path]]],
    task_types: Optional[Sequence[str]] = None,
    difficulties: Optional[Sequence[str]] = None,
    max_samples_per_file: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 111,
) -> MQAJsonlDataset:
    """从一个或多个JSONL文件路径构建数据集
    
    如果提供多个路径，则它们会在内存中合并到临时数据集对象中。
    
    Args:
        jsonl_paths: JSONL文件路径（可以是一个或多个）
        task_types: 限制的任务类型列表
        difficulties: 限制的难度级别列表
        max_samples_per_file: 每个文件的最大样本数
        shuffle: 是否打乱数据
        seed: 随机种子
        
    Returns:
        MQAJsonlDataset: 合并后的数据集
    """

    if isinstance(jsonl_paths, (str, Path)):
        return MQAJsonlDataset(
            jsonl_paths,
            task_types=task_types,
            difficulties=difficulties,
            max_samples=max_samples_per_file,
            shuffle=shuffle,
            seed=seed,
        )

    merged_rows: List[Dict[str, Any]] = []
    for path in jsonl_paths:
        ds = MQAJsonlDataset(
            path,
            task_types=task_types,
            difficulties=difficulties,
            max_samples=max_samples_per_file,
            shuffle=False,  # 单个文件不需要打乱，最后统一打乱
            seed=seed,
        )
        merged_rows.extend(ds.examples)  # 合并示例

    temp = object.__new__(MQAJsonlDataset)  # 创建空的MQAJsonlDataset实例
    temp.jsonl_path = ",".join(str(p) for p in jsonl_paths)  # 记录所有文件路径
    temp.task_types = _ensure_list(task_types)
    temp.difficulties = _ensure_list(difficulties)
    temp.max_samples = None
    temp.shuffle = shuffle
    temp.seed = seed
    temp.require_score = False
    temp.examples = merged_rows  # 设置合并后的示例
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(temp.examples)  # 打乱合并后的数据
    return temp


def split_dataset(
    dataset: MQAJsonlDataset,
    train_ratio: float = 0.8,
    seed: int = 111,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """分割数据集为训练集和验证集
    
    Args:
        dataset: 要分割的数据集
        train_ratio: 训练集所占比例
        seed: 随机种子
        
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: 训练集和验证集的元组
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")  # 检查训练比例是否合理
    examples = list(dataset.examples)  # 复制示例列表
    rng = random.Random(seed)
    rng.shuffle(examples)  # 随机打乱数据
    cut = int(len(examples) * train_ratio)  # 计算分割点
    return examples[:cut], examples[cut:]  # 返回训练集和验证集


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """将批次数据移动到指定设备
    
    Args:
        batch: 包含张量和其他数据的批次字典
        device: 目标设备
        
    Returns:
        Dict[str, Any]: 移动到目标设备的批次数据
    """
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)  # 将张量移到指定设备
        else:
            moved[key] = value  # 非张量保持不变
    return moved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for MQA dataloader.")  # 创建命令行参数解析器
    parser.add_argument("--jsonl", type=str, required=True, help="Path to mqa_rule_choice.jsonl or mqa_score_band.jsonl")  # 添加JSONL文件路径参数
    parser.add_argument("--max_samples", type=int, default=3)  # 添加最大样本数参数
    args = parser.parse_args()

    ds = MQAJsonlDataset(args.jsonl, max_samples=args.max_samples)  # 创建数据集实例
    print(json.dumps(ds.get_stats(), ensure_ascii=False, indent=2))  # 打印数据集统计信息
    if len(ds) > 0:
        preview = ds[0].copy()
        preview.pop("choice_texts", None)  # 从预览中移除choice_texts
        print(json.dumps(preview, ensure_ascii=False, indent=2))  # 打印预览信息