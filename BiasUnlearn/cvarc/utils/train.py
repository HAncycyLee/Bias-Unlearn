train.py
from __future__ import annotations

"""Minimal viable trainer for MQA rule_choice sequence classification.

此脚本设计用于与`dataloader_mqa.py`和`build_mqa_from_json.py`生成的JSONL工件一起工作。
初始关注点有意保持狭窄：

- 任务: rule_choice
- 公式化: 序列分类
- 骨干: AutoModelForSequenceClassification
- 数据: 一个JSONL文件或明确的训练/评估JSONL文件

关键特性:
- 通过MQAJsonlDataset读取MQA JSONL
- 使用序列分类整理器
- 支持从单个JSONL进行训练/评估拆分
- 计算准确率+宏F1分数
- 将指标记录到metrics.jsonl
- 按评估准确性保存最佳检查点
- 可选的LoRA用于7B模型的高效微调

示例: 参考同文件夹下train.sh
"""

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from dataloader_mqa import (
    MQAJsonlDataset,
    MQASequenceClassificationCollator,
    batch_to_device,
)

# 定义规则选择任务的标签数量和映射
RULE_CHOICE_NUM_LABELS = 4
LABEL_ID_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

@dataclass
class TrainConfig:
    """训练配置数据类
    
    包含所有训练过程中的配置参数
    """

    train_jsonl: str  # 训练数据JSONL文件路径
    eval_jsonl: Optional[str]  # 评估数据JSONL文件路径（可选）
    model_name_or_path: str  # 模型名称或路径
    output_dir: str  # 输出目录
    task_type: str = "rule_choice"  # 任务类型，默认为rule_choice
    max_length: int = 1024  # 最大序列长度，默认1024
    train_split_ratio: float = 0.85  # 训练集分割比例，默认0.85
    seed: int = 123  # 随机种子，默认123
    per_device_train_batch_size: int = 2  # 每设备训练批次大小，默认2
    per_device_eval_batch_size: int = 4  # 每设备评估批次大小，默认4
    gradient_accumulation_steps: int = 1  # 梯度累积步数，默认1
    learning_rate: float = 2e-5  # 学习率，默认2e-5
    weight_decay: float = 0.01  # 权重衰减，默认0.01
    num_train_epochs: int = 3  # 训练轮次，默认3
    warmup_ratio: float = 0.05  # 预热比例，默认0.05
    max_grad_norm: float = 1.0  # 最大梯度范数，默认1.0
    logging_steps: int = 10  # 日志记录步数，默认10
    save_every_epoch: bool = True  # 是否每轮都保存模型，默认True
    difficulties: Optional[List[str]] = None  # 难度等级列表（可选）
    max_train_samples: Optional[int] = None  # 最大训练样本数（可选）
    max_eval_samples: Optional[int] = None  # 最大评估样本数（可选）
    num_workers: int = 0  # 数据加载进程数，默认0
    bf16: bool = False  # 是否使用bfloat16精度，默认False
    fp16: bool = False  # 是否使用float16精度，默认False
    use_lora: bool = False  # 是否使用LoRA微调，默认False
    load_in_4bit: bool = False  # 是否4位量化加载，默认False
    lora_r: int = 16  # LoRA秩参数，默认16
    lora_alpha: int = 32  # LoRA缩放参数，默认32
    lora_dropout: float = 0.05  # LoRA dropout比率，默认0.05
    lora_target_modules: Optional[List[str]] = None  # LoRA目标模块列表（可选）


def parse_args() -> TrainConfig:
    """解析命令行参数并返回训练配置对象
    
    Returns:
        TrainConfig: 包含所有训练参数的配置对象
        
    Raises:
        ValueError: 当参数不符合要求时抛出异常
    """

    parser = argparse.ArgumentParser(description="Train MQA sequence-classification model for rule_choice.")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_type", type=str, default="rule_choice")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--train_split_ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--difficulties", nargs="*", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated target modules, e.g. q_proj,k_proj,v_proj,o_proj",
    )
    args = parser.parse_args()

    if args.task_type != "rule_choice":
        raise ValueError("This minimal trainer currently only supports --task_type rule_choice")
    if args.train_split_ratio <= 0.0 or args.train_split_ratio >= 1.0:
        raise ValueError("--train_split_ratio must be in (0, 1)")
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 or --fp16")
    if args.load_in_4bit and not args.use_lora:
        raise ValueError("--load_in_4bit is only supported together with --use_lora in this script")

    target_modules = None
    if args.lora_target_modules:
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

    return TrainConfig(
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        task_type=args.task_type,
        max_length=args.max_length,
        train_split_ratio=args.train_split_ratio,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_every_epoch=args.save_every_epoch,
        difficulties=args.difficulties,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_workers=args.num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        use_lora=args.use_lora,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=target_modules,
    )

def set_seed(seed: int) -> None:
    """设置随机种子以确保实验可重现
    
    Args:
        seed: 随机种子值
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        Path: 创建的目录路径对象
    """

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_metric(output_dir: Path, payload: Dict[str, Any]) -> None:
    """追加指标到日志文件
    
    Args:
        output_dir: 输出目录路径
        payload: 要写入的指标数据
    """
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """保存JSON数据到指定路径
    
    Args:
        path: 保存路径
        payload: 要保存的数据
    """

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class DatasetView(Dataset):
    """数据集视图类，提供对示例列表的访问接口"""
    
    def __init__(self, examples: List[Dict[str, Any]]):
        """初始化数据集视图
        
        Args:
            examples: 示例列表
        """

        self.examples = examples

    def __len__(self) -> int:
        """获取数据集长度
        
        Returns:
            int: 数据集中示例的数量
        """

        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取指定索引的示例
        
        Args:
            index: 索引
            
        Returns:
            Dict[str, Any]: 对应索引的示例数据
        """

        return self.examples[index]


def split_examples(
    examples: List[Dict[str, Any]],
    train_ratio: float,
    seed: int,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按标签分布分割训练和评估示例
    
    Args:
        examples: 输入示例列表
        train_ratio: 训练集比例
        seed: 随机种子
        max_train_samples: 最大训练样本数（可选）
        max_eval_samples: 最大评估样本数（可选）
        
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: 训练集和评估集示例列表
    """

    # 按标签分组示例
    by_label: Dict[int, List[Dict[str, Any]]] = {}
    for ex in examples:
        by_label.setdefault(int(ex["label"]), []).append(ex) # 添加示例到对应标签的列表中

    rng = random.Random(seed)
    train_examples: List[Dict[str, Any]] = []
    eval_examples: List[Dict[str, Any]] = []

    # 为每个标签随机分割训练和评估示例
    for label, rows in by_label.items():
        rows = list(rows)
        rng.shuffle(rows)
        # 确保至少有一个示例用于评估，除非只有一条记录
        cut = max(1, int(len(rows) * train_ratio)) if len(rows) > 1 else len(rows)
        if cut >= len(rows) and len(rows) > 1:
            cut = len(rows) - 1
        train_examples.extend(rows[:cut])
        eval_examples.extend(rows[cut:])

    rng.shuffle(train_examples)
    rng.shuffle(eval_examples)

    # 如果设置了最大样本数限制，则裁剪数据集
    if max_train_samples is not None:
        train_examples = train_examples[:max_train_samples]
    if max_eval_samples is not None:
        eval_examples = eval_examples[:max_eval_samples]
    return train_examples, eval_examples


def load_datasets(cfg: TrainConfig) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    """根据配置加载训练和评估数据集
    
    Args:
        cfg: 训练配置对象
        
    Returns:
        Tuple[Dataset, Dataset, Dict[str, Any]]: 训练集、评估集和统计数据
    """

    train_ds_all = MQAJsonlDataset(
        cfg.train_jsonl,
        task_types=[cfg.task_type],
        difficulties=cfg.difficulties,
        shuffle=False,
    )
    if len(train_ds_all) == 0:
        raise ValueError(f"No examples found in {cfg.train_jsonl} after filtering")

    # 如果提供了评估JSONL路径，则直接加载评估集
    if cfg.eval_jsonl:
        eval_ds = MQAJsonlDataset(
            cfg.eval_jsonl,
            task_types=[cfg.task_type],
            difficulties=cfg.difficulties,
            max_samples=cfg.max_eval_samples,
            shuffle=False,
        )
        train_examples = list(train_ds_all.examples)
        if cfg.max_train_samples is not None:
            train_examples = train_examples[: cfg.max_train_samples]
        train_ds = DatasetView(train_examples)
        stats = {
            "train": train_ds_all.get_stats(),
            "eval": eval_ds.get_stats(),
            "split": "explicit_eval_jsonl",
        }
        return train_ds, eval_ds, stats

    # 否则从训练数据中分割出训练集和评估集
    train_examples, eval_examples = split_examples(
        train_ds_all.examples,
        train_ratio=cfg.train_split_ratio,
        seed=cfg.seed,
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
    )
    if not eval_examples:
        raise ValueError("Eval split is empty. Provide --eval_jsonl or use a larger dataset.")

    train_ds = DatasetView(train_examples)
    eval_ds = DatasetView(eval_examples)
    stats = {
        "train": {
            "num_examples": len(train_examples),
            "task_distribution": {cfg.task_type: len(train_examples)},
        },
        "eval": {
            "num_examples": len(eval_examples),
            "task_distribution": {cfg.task_type: len(eval_examples)},
        },
        "split": "stratified_from_single_jsonl",
        "source_jsonl": cfg.train_jsonl,
    }
    return train_ds, eval_ds, stats


def resolve_dtype(cfg: TrainConfig) -> Optional[torch.dtype]:
    """根据配置解析数据类型
    
    Args:
        cfg: 训练配置对象
        
    Returns:
        Optional[torch.dtype]: PyTorch数据类型，如果没有指定则返回None
    """

    if cfg.bf16:
        return torch.bfloat16
    if cfg.fp16:
        return torch.float16
    return None


def load_model_and_tokenizer(cfg: TrainConfig):
    """根据配置加载模型和分词器
    
    Args:
        cfg: 训练配置对象
        
    Returns:
        tuple: 模型和分词器
    """

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = None
    model_kwargs: Dict[str, Any] = {
        "num_labels": RULE_CHOICE_NUM_LABELS,
        "trust_remote_code": True,
    }
    dtype = resolve_dtype(cfg)
    if dtype is not None and not cfg.load_in_4bit:
        model_kwargs["torch_dtype"] = dtype
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype or torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    # 此处加载的是线性模型，后面是不是不该这么弄？
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("peft is required for --use_lora") from exc

        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        target_modules = cfg.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    return model, tokenizer

def create_optimizer(
    model: torch.nn.Module, 
    lr: float, 
    weight_decay: float
    ) -> AdamW:
    """创建优化器，对不同类型的参数应用不同的权重衰减
    
    Args:
        model: PyTorch模型
        lr: 学习率
        weight_decay: 权重衰减系数
        
    Returns:
        AdamW: 优化器实例
    """
    
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: # 不需要优化的参数
            continue
        # 对于一维参数、偏置项或归一化层不应用权重衰减
        if param.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr)


def compute_macro_f1(preds: Sequence[int], labels: Sequence[int], num_labels: int) -> float:
    """计算宏观F1分数
    
    Args:
        preds: 预测结果序列
        labels: 真实标签序列
        num_labels: 标签总数
        
    Returns:
        float: 宏观F1分数
    """

    f1s: List[float] = []
    for cls in range(num_labels):
        # 计算每个类别的TP, FP, FN
        tp = sum(1 for p, y in zip(preds, labels) if p == cls and y == cls)
        fp = sum(1 for p, y in zip(preds, labels) if p == cls and y != cls)
        fn = sum(1 for p, y in zip(preds, labels) if p != cls and y == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def compute_accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    """计算准确率
    
    Args:
        preds: 预测结果序列
        labels: 真实标签序列
        
    Returns:
        float: 准确率
    """

    if not labels:
        return 0.0
    return sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """评估模型性能
    
    Args:
        model: 要评估的模型
        dataloader: 数据加载器
        device: 计算设备
        
    Returns:
        Dict[str, Any]: 包含损失、准确率、宏观F1分数等指标的字典
    """

    model.eval()
    total_loss = 0.0
    total_examples = 0
    preds: List[int] = []
    labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            logits = outputs.logits
            batch_size = batch["labels"].size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())

    avg_loss = total_loss / max(total_examples, 1)
    acc = compute_accuracy(preds, labels)
    macro_f1 = compute_macro_f1(preds, labels, num_labels=RULE_CHOICE_NUM_LABELS)
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "num_examples": total_examples,
    }


def save_checkpoint(model, tokenizer, output_dir: Path, name: str) -> None:
    """保存模型检查点
    
    Args:
        model: 模型
        tokenizer: 分词器
        output_dir: 输出目录
        name: 检查点名称
    """

    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


def train(cfg: TrainConfig) -> None:
    """执行训练流程
    
    Args:
        cfg: 训练配置对象
    """

    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)
    save_json(output_dir / "train_config.json", asdict(cfg))

    model, tokenizer = load_model_and_tokenizer(cfg)
    collator = MQASequenceClassificationCollator(tokenizer=tokenizer, max_length=cfg.max_length)
    train_ds, eval_ds, dataset_stats = load_datasets(cfg)
    save_json(output_dir / "dataset_stats.json", dataset_stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.load_in_4bit:
        model.to(device)

    optimizer = create_optimizer(model, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_train_steps = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps) * cfg.num_train_epochs
    warmup_steps = int(total_train_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    use_amp = cfg.fp16 or cfg.bf16
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16 and torch.cuda.is_available())

    best_eval_acc = -1.0
    global_step = 0

    # 记录模型参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    append_metric(
        output_dir,
        {
            "event": "run_start",
            "task": cfg.task_type,
            "model_name_or_path": cfg.model_name_or_path,
            "num_train_examples": len(train_ds),
            "num_eval_examples": len(eval_ds),
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "trainable_ratio": float(trainable_params / max(total_params, 1)),
        },
    )

    # 1. 训练循环
    for epoch in range(1, cfg.num_train_epochs + 1):
        model.train() # 将模型设置为训练模式（启用了Dropout和BatchNorm）
        running_loss = 0.0
        epoch_examples = 0
        optimizer.zero_grad(set_to_none=True) # 清空梯度，set_to_none=True 可以节省内存并可能提高性能

        # 2. Batch数据处理和混合精度预测
        for step, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, device) # 将数据移动到设备上（GPU/CPU）
            batch_size = batch["labels"].size(0)

            # 自动混合精度（AMP）开启上下文
            # AMP (autocast)：通过使用 FP16 或 BF16 精度进行计算
            # 可以显著减少显存占用并加快计算速度
            with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available(), dtype=amp_dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                # 损失除以梯度累积步数
                # 为了模拟更大的 Batch Size，损失被除以累积步数，
                # 这样多次反向传播累加后的梯度才等于大 Batch 的梯度均值
                loss = outputs.loss / cfg.gradient_accumulation_steps

            # 3. 反向传播
            # 在使用混合精度时，由于 FP16 表示范围有限，微小的梯度可能会变成 0。
            # GradScaler 会放大 Loss 以保留这些梯度。
            if scaler.is_enabled():
                scaler.scale(loss).backward() # 放大loss防止梯度下溢
            else:
                loss.backward()

            running_loss += float(loss.item()) * cfg.gradient_accumulation_steps * batch_size
            epoch_examples += batch_size

            # 4. 优化器更新（梯度积累和裁剪）
            if step % cfg.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer) # 反向缩放梯度
                # 梯度裁剪，防止梯度爆炸
                # 强制梯度的模长不超过 max_grad_norm，这
                # 是训练 Transformer 模型时的常用技巧，保证训练稳定性
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer) # 更新参数
                    scaler.update() # 更新缩放因子
                else:
                    # 只有当经过 cfg.gradient_accumulation_steps 个 batch 后
                    # 才真正执行一次参数更新（optimizer.step()）
                    optimizer.step()
                scheduler.step() # 更新学习率
                optimizer.zero_grad(set_to_none=True)
                global_step += 1 # 记录总共进行了多少次参数更新，而不是处理了多少个 batch

                # 5. 记录训练进度
                # 在训练过程中定期记录当前的损失值和学习率，便于
                # 通过 TensorBoard 或日志文件观察收敛情况
                if cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                    append_metric(
                        output_dir,
                        {
                            "event": "train_step",
                            "epoch": epoch,
                            "global_step": global_step,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "running_train_loss": running_loss / max(epoch_examples, 1),
                        },
                    )

        # 6. Epoch 结束评估与模型保存
        train_loss = running_loss / max(epoch_examples, 1)
        eval_metrics = evaluate(model, eval_loader, device=device) # 执行评估逻辑

        epoch_payload = {
            "event": "epoch_end",
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_macro_f1": eval_metrics["macro_f1"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        append_metric(output_dir, epoch_payload)
        print(json.dumps(epoch_payload, ensure_ascii=False))

        # 如果配置要求，每轮都保存
        if cfg.save_every_epoch:
            save_checkpoint(model, tokenizer, output_dir, f"checkpoint-epoch-{epoch}")

        # 如果当前准确率是历史上最好的，保存 "best_checkpoint"
        if eval_metrics["accuracy"] > best_eval_acc:
            best_eval_acc = eval_metrics["accuracy"]
            save_checkpoint(model, tokenizer, output_dir, "best_checkpoint")
            save_json(
                output_dir / "best_metrics.json",
                {
                    "best_epoch": epoch,
                    **eval_metrics,
                },
            )

    append_metric(
        output_dir,
        {
            "event": "run_end",
            "best_eval_accuracy": best_eval_acc,
        },
    )


def main() -> None:
    """主函数，执行整个训练流程"""
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()