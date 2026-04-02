"""
DataLoader模块用于处理StereoSet数据集，提供对刻板印象数据的加载和预处理功能。
包含多个数据加载器类，用于处理不同类型的偏见检测任务。
"""

# encoding:utf-8
#    Date  :  2025/4/2
# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random
import re
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, RandomSampler

# 设置随机种子以确保实验可复现
torch.manual_seed(111)
np.random.seed(111)
random.seed(111)
import json
import string
from tqdm import tqdm


class SentimentIntrasentenceLoader(object):
    """
    情感内句数据加载器，用于处理内句情感分析数据
    
    该加载器从StereoSet数据集中提取内句示例，并为模型训练准备输入数据
    """
    def __init__(self, tokenizer, max_seq_length=None, pad_to_max_length=False, input_file="../../data/bias.json"):
        """
        初始化SentimentIntrasentenceLoader
        
        参数:
            tokenizer: 分词器，用于文本编码
            max_seq_length: 最大序列长度，超过此长度将被截断
            pad_to_max_length: 是否填充到最大长度
            input_file: 输入文件路径，默认为"../../data/bias.json"
        """
        # 加载StereoSet数据
        stereoset = StereoSet(input_file)
        # 获取内句示例
        clusters = stereoset.get_intrasentence_examples()
        
        self.tokenizer = tokenizer
        self.sentences = []
        self.MASK_TOKEN = self.tokenizer.mask_token
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length

        # 如果使用XLNet分词器，需要添加前缀文本
        if tokenizer.__class__.__name__ == "XLNetTokenizer":
            self.prepend_text = """In 1991, the remains of Russian Tsar Nicholas II and his family
            (except for Alexei and Maria) are discovered.
            The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
            remainder of the story. 1883 Western Siberia,
            a young Grigori Rasputin is asked by his father and a group of men to perform magic.
            Rasputin has a vision and denounces one of the men as a horse thief. Although his
            father initially slaps him for making such an accusation, Rasputin watches as the
            man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
            the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
            with people, even a bishop, begging for his blessing. <eod> """

        # 遍历所有集群和句子，构建输入数据
        for cluster in clusters:
            for sentence in cluster.sentences:
                # 将上下文中的BLANK替换为模板词
                new_sentence = cluster.context.replace("BLANK", sentence.template_word)
                self.sentences.append((new_sentence, sentence.ID))

    def __len__(self):
        """返回句子总数"""
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        
        参数:
            idx: 数据索引
            
        返回:
            tuple: 包含句子ID、输入ID、注意力掩码和token类型ID的元组
        """
        sentence, sentence_id = self.sentences[idx]
        if self.tokenizer.__class__.__name__ == "XLNetTokenizer":
            text = self.prepend_text
            text_pair = sentence
        else:
            text = sentence
            text_pair = None
        
        # 使用分词器编码文本
        tokens_dict = self.tokenizer.encode_plus(
            text, 
            text_pair=text_pair, 
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=self.pad_to_max_length, 
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=False, 
            return_special_tokens_mask=False,
            return_tensors="pt"
        )
        
        input_ids = tokens_dict['input_ids']
        attention_mask = tokens_dict['attention_mask']
        token_type_ids = tokens_dict['token_type_ids']
        return sentence_id, input_ids, attention_mask, token_type_ids


class IntrasentenceLoader(object):
    """
    内句数据加载器，用于处理内句偏见检测任务
    
    该加载器生成掩码语言建模任务的训练数据
    """
    def __init__(self, tokenizer, max_seq_length=None, pad_to_max_length=False, input_file="../../data/bias.json"):
        """
        初始化IntrasentenceLoader
        
        参数:
            tokenizer: 分词器，用于文本编码
            max_seq_length: 最大序列长度，超过此长度将被截断
            pad_to_max_length: 是否填充到最大长度
            input_file: 输入文件路径，默认为"../../data/bias.json"
        """
        # 加载StereoSet数据
        stereoset = StereoSet(input_file)
        clusters = stereoset.get_intrasentence_examples()
        
        self.tokenizer = tokenizer
        self.sentences = []
        self.MASK_TOKEN = self.tokenizer.mask_token
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length

        # 如果使用XLNet分词器，需要添加前缀文本
        if tokenizer.__class__.__name__ == "XLNetTokenizer":
            self.prepend_text = """In 1991, the remains of Russian Tsar Nicholas II and his family
            (except for Alexei and Maria) are discovered.
            The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
            remainder of the story. 1883 Western Siberia,
            a young Grigori Rasputin is asked by his father and a group of men to perform magic.
            Rasputin has a vision and denounces one of the men as a horse thief. Although his
            father initially slaps him for making such an accusation, Rasputin watches as the
            man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
            the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
            with people, even a bishop, begging for his blessing. <eod> """

        # 构建输入数据，将模板词逐个token插入到上下文中
        for cluster in clusters:
            for sentence in cluster.sentences:
                # 将模板词分解为多个token
                insertion_tokens = self.tokenizer.encode(sentence.template_word, add_special_tokens=False)
                for idx in range(len(insertion_tokens)):
                    # 创建插入字符串，使用MASK_TOKEN作为占位符
                    insertion = self.tokenizer.decode(insertion_tokens[:idx])
                    insertion_string = f"{insertion}{self.MASK_TOKEN}"
                    # 将上下文中的BLANK替换为插入字符串
                    new_sentence = cluster.context.replace("BLANK", insertion_string)
                    next_token = insertion_tokens[idx]  # 下一个应该预测的token
                    self.sentences.append((new_sentence, sentence.ID, next_token))

    def __len__(self):
        """返回句子总数"""
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        
        参数:
            idx: 数据索引
            
        返回:
            tuple: 包含句子ID、下一个token、输入ID、注意力掩码和token类型ID的元组
        """
        sentence, sentence_id, next_token = self.sentences[idx]
        if self.tokenizer.__class__.__name__ == "XLNetTokenizer":
            text = self.prepend_text
            text_pair = sentence
        else:
            text = sentence
            text_pair = None
        
        # 使用分词器编码文本
        tokens_dict = self.tokenizer.encode_plus(
            text, 
            text_pair=text_pair, 
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=self.pad_to_max_length, 
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=False, 
            return_special_tokens_mask=False
        )
        
        input_ids = tokens_dict['input_ids']
        attention_mask = tokens_dict['attention_mask']
        token_type_ids = tokens_dict['token_type_ids']
        return sentence_id, next_token, input_ids, attention_mask, token_type_ids


class StereoSet(object):
    """
    StereoSet数据集类，用于加载和处理偏见检测数据
    
    该类提供了访问内句和跨句示例的方法
    """
    def __init__(self, location, json_obj=None):
        """
        初始化StereoSet对象

        参数
        ----------
        location (string): StereoSet.json文件的位置
        json_obj: JSON对象，如果提供则直接使用该对象而不是从文件读取
        """
        if json_obj == None:
            with open(location, "r") as f:
                self.json = json.load(f)
        else:
            self.json = json_obj

        self.version = self.json['version']
        # 创建内句示例
        self.intrasentence_examples = self.__create_intrasentence_examples__(
            self.json['data']['intrasentence'])
        # 创建跨句示例
        self.intersentence_examples = self.__create_intersentence_examples__(
            self.json['data']['intersentence'])

    def __create_intrasentence_examples__(self, examples):
        """
        创建内句示例

        参数
        ----------
        examples: 内句示例列表

        返回
        -------
        created_examples: 创建的内句示例列表
        """
        created_examples = []
        count={}
        for example in examples:
            sentences = []
            for sentence in example['sentences']:
                labels = []
                for label in sentence['labels']:
                    labels.append(Label(**label))
                
                # 创建句子对象
                sentence_obj = Sentence(
                    sentence['id'], sentence['sentence'], labels, sentence['gold_label'])
                
                # 找到上下文中的BLANK位置
                word_idx = None
                for idx, word in enumerate(example['context'].split(" ")):
                    if "BLANK" in word:
                        word_idx = idx
                if word_idx is None:
                    raise Exception("No blank word found.")
                
                # 提取模板词（不包含标点符号）
                template_word = sentence['sentence'].split(" ")[word_idx]
                sentence_obj.template_word = template_word.translate(str.maketrans('', '', string.punctuation))
                sentences.append(sentence_obj)

            # 创建内句示例对象
            created_example = IntrasentenceExample(
                example['id'], example['bias_type'],
                example['target'], example['context'], sentences)
            
            # 控制每个偏见类型的最大示例数量为100
            if example['bias_type'] not in count:
                count[example['bias_type']]=0
            if count[example['bias_type']]<100:
                count[example['bias_type']]+=1
                created_examples.append(created_example)
        return created_examples

    def __create_intersentence_examples__(self, examples):
        """
        创建跨句示例

        参数
        ----------
        examples: 跨句示例列表

        返回
        -------
        created_examples: 创建的跨句示例列表
        """
        created_examples = []
        for example in examples:
            sentences = []
            for sentence in example['sentences']:
                labels = []
                for label in sentence['labels']:
                    labels.append(Label(**label))
                sentence = Sentence(
                    sentence['id'], sentence['sentence'], labels, sentence['gold_label'])
                sentences.append(sentence)
            
            # 创建跨句示例对象
            created_example = IntersentenceExample(
                example['id'], example['bias_type'], example['target'],
                example['context'], sentences)
            created_examples.append(created_example)
        return created_examples

    def get_intrasentence_examples(self):
        """获取内句示例"""
        return self.intrasentence_examples

    def get_intersentence_examples(self):
        """获取跨句示例"""
        return self.intersentence_examples


class Example(object):
    """
    示例基类，表示一个通用的示例对象
    """
    def __init__(self, ID, bias_type, target, context, sentences):
        """
         初始化示例对象

         参数
         ----------
         ID (string): 提供示例的唯一ID
         bias_type (string): 偏见类型的描述，必须是[RACE, RELIGION, GENDER, PROFESSION]之一
         target (string): 被刻板化的词
         context (string): 上下文句子，如果存在，设置刻板印象的背景
         sentences (list): 与目标相关的句子列表
         """
        self.ID = ID
        self.bias_type = bias_type
        self.target = target
        self.context = context
        self.sentences = sentences

    def __str__(self):
        """返回示例的字符串表示"""
        s = f"Domain: {self.bias_type} - Target: {self.target} \r\n"
        s += f"Context: {self.context} \r\n"
        for sentence in self.sentences:
            s += f"{sentence} \r\n"
        return s


class Sentence(object):
    """
    句子类，表示一个句子对象
    """
    def __init__(self, ID, sentence, labels, gold_label):
        """
        初始化句子对象

        参数
        ----------
        ID (string): 提供相对于示例的句子唯一ID
        sentence (string): 文本句子
        labels (list of Label objects): 句子的人工标签列表
        gold_label (enum): 与句子关联的黄金标签，通过标签的argmax计算得出
            必须是[stereotype, anti-stereotype, unrelated]之一
        """
        assert type(ID) == str
        assert gold_label in ['stereotype', 'anti-stereotype', 'unrelated']
        assert isinstance(labels, list)
        assert isinstance(labels[0], Label)

        self.ID = ID
        self.sentence = sentence
        self.gold_label = gold_label
        self.labels = labels
        self.template_word = None

    def __str__(self):
        """返回句子的字符串表示"""
        return f"{self.gold_label.capitalize()} Sentence: {self.sentence}"


class Label(object):
    """
    标签类，表示一个标签对象
    """
    def __init__(self, human_id, label):
        """
        初始化标签对象

        参数
        ----------
        human_id (string): 提供标记句子的人的唯一ID
        label (enum): 提供句子的标签，必须是[stereotype, anti-stereotype, unrelated]之一
        """
        assert label in ['stereotype',
                         'anti-stereotype', 'unrelated', 'related']
        self.human_id = human_id
        self.label = label


class IntrasentenceExample(Example):
    """
    内句示例类，继承自Example类
    """
    def __init__(self, ID, bias_type, target, context, sentences):
        """
        实现内句示例的Example类

        参考Example的文档字符串获取更多信息
        """
        super(IntrasentenceExample, self).__init__(
            ID, bias_type, target, context, sentences)


class IntersentenceExample(Example):
    """
    跨句示例类，继承自Example类
    """
    def __init__(self, ID, bias_type, target, context, sentences):
        """
        实现跨句示例的Example类

        参考Example的文档字符串获取更多信息
        """
        super(IntersentenceExample, self).__init__(
            ID, bias_type, target, context, sentences)


def get_stereoset_answers_plaintext(
    tokenizer, 
    ster_batch_size=4, 
    batch_size=28, 
    mix_anti=False, 
    exclude=[], 
    file_path="StereoSet/test.json", 
    type2id=None
):
    """
    从StereoSet数据集中获取原始文本格式的答案
    
    该函数加载数据集并创建不同类型偏见的数据加载器
    
    参数:
        tokenizer: 分词器
        ster_batch_size: 刻板印象数据的批处理大小
        batch_size: 其他数据的批处理大小
        mix_anti: 是否混合反刻板印象数据
        exclude: 排除的偏见类型列表
        file_path: 数据文件路径
        type2id: 偏见类型到ID的映射
    
    返回:
        多个数据加载器，分别对应不同类型的数据
    """
    anti=[]
    ster=[]
    unrelate=[]
    ster_race, ster_gender, ster_profession, ster_religion = [], [], [], []
    anti_race, anti_gender, anti_profession, anti_religion = [], [], [], []
    
    # 按偏见类型组织数据
    category_map = {
        "stereotype": {
            "race": ster_race,
            "gender": ster_gender,
            "profession": ster_profession,
            "religion": ster_religion
        },
        "anti-stereotype": {
            "race": anti_race,
            "gender": anti_gender,
            "profession": anti_profession,
            "religion": anti_religion
        }
    }

    # 从文件加载数据
    with open(file_path) as fr:
        objs=json.load(fr)
        # 随机打乱数据顺序
        random.shuffle(objs["data"]["intersentence"])
        random.shuffle(objs["data"]["intrasentence"])
        
        # 处理跨句数据
        for obj in objs["data"]["intersentence"]:
            bias_type = obj["bias_type"]
            for sen in obj["sentences"]:
                if bias_type in exclude:
                    continue

                # 组合上下文和句子
                combined = (obj["context"], obj["context"] + ' ' + sen["sentence"], type2id[obj["bias_type"]])
                
                # 根据标签类型将数据分配到不同的列表中
                if sen["gold_label"] == "anti-stereotype":
                    if bias_type == "race":
                        anti_race.append(combined)
                    elif bias_type == "gender":
                        anti_gender.append(combined)
                    elif bias_type == "profession":
                        anti_profession.append(combined)
                    elif bias_type == "religion":
                        anti_religion.append(combined)

                elif sen["gold_label"] == "stereotype":
                    if bias_type == "race":
                        ster_race.append(combined)
                    elif bias_type == "gender":
                        ster_gender.append(combined)
                    elif bias_type == "profession":
                        ster_profession.append(combined)
                    elif bias_type == "religion":
                        ster_religion.append(combined)
                else:
                    unrelate.append(combined)

        print("len of anti,ster,unrelate:",len(anti),len(ster),len(unrelate))

        # 处理内句数据
        for obj in objs["data"]["intrasentence"]:
            # 过滤掉包含多个BLANK或特殊字符的上下文
            if len(re.findall("BLANK", obj["context"]))>1:
                continue
            elif ".BLANK" in obj["context"] or "`BLANK" in obj["context"]:
                continue

            # 提取上下文部分
            context = re.split("\s?BLANK", obj["context"])[0]
            bias_type = obj["bias_type"]
            
            for sen in obj["sentences"]:
                if bias_type in exclude:
                    continue
                
                # 组合上下文和句子
                combined = (context, sen["sentence"], type2id[obj["bias_type"]])
                
                # 根据标签类型将数据分配到不同的列表中
                if sen["gold_label"] == "anti-stereotype":
                    if bias_type == "race":
                        anti_race.append(combined)
                    elif bias_type == "gender":
                        anti_gender.append(combined)
                    elif bias_type == "profession":
                        anti_profession.append(combined)
                    elif bias_type == "religion":
                        anti_religion.append(combined)
                elif sen["gold_label"] == "stereotype":
                    if bias_type == "race":
                        ster_race.append(combined)
                    elif bias_type == "gender":
                        ster_gender.append(combined)
                    elif bias_type == "profession":
                        ster_profession.append(combined)
                    elif bias_type == "religion":
                        ster_religion.append(combined)
                else:
                    unrelate.append(combined)

        print('lenlenlenlen')
        for k,v in category_map["stereotype"].items():
            print(k,len(v))

        # 合并不同类型的刻板印象和反刻板印象数据
        ster = ster_gender + ster_profession + ster_race[:len(ster_race)] + ster_religion[:len(ster_religion)]
        anti = anti_gender + anti_profession + anti_race[:len(ster_race)] + anti_religion[:len(ster_religion)]
        print("len of anti,ster,unrelate:",len(anti),len(ster),len(unrelate))
        
        # 如果启用混合反刻板印象数据
        if mix_anti:
            ster+=anti[:len(anti)//4]

            ster_gender+= anti_gender[:len(anti_gender)//4]
            ster_profession+= anti_profession[:len(anti_profession)//4]
            ster_race+= anti_race[:len(anti_race)//4]
            ster_religion+= anti_religion[:len(anti_religion)//4]


    def get_dataloader(tuple_data, batch_size):
        """
        创建数据加载器
        
        参数:
            tuple_data: 包含(上下文, 文本, 偏见类型)元组的数据列表
            batch_size: 批处理大小
            
        返回:
            DataLoader: 数据加载器对象
        """
        # 初始化数据字典
        data = {"input_ids": [], "attention_mask": [], "start_locs":[],"bias_type":[]}
        
        # 遍历数据，进行编码和处理
        for context, text,  bias_type in tqdm(tuple_data):
            # 根据分词器类型添加开始标记
            if "gpt" in tokenizer.__class__.__name__.lower():
                text = tokenizer.bos_token + text
            
            # 编码文本
            tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=100)
            
            # 添加编码后的输入ID和注意力掩码
            data["input_ids"].append(tokenized["input_ids"])
            data["attention_mask"].append(tokenized["attention_mask"])
            data["bias_type"].append(bias_type)
            
            # 获取偏移映射以确定上下文结束位置
            offset_mapping = tokenized["offset_mapping"]

            context_len = len(context)
            
            # 找到上下文在token序列中的结束位置
            for idx, (start, end) in enumerate(offset_mapping):
                if end > context_len:
                    data["start_locs"].append(idx)
                    break

        # 创建数据集
        dataset = Dataset.from_dict(data)
        # 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        # 创建随机采样器
        sampler = RandomSampler(dataset)
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=4, 
            collate_fn=data_collator, 
            sampler=sampler
        )
        return dataloader

    # 为不同类型的数据创建数据加载器
    anti_dataloader=get_dataloader(anti, batch_size)
    ster_dataloader=get_dataloader(ster, ster_batch_size)
    unrelate_dataloader=get_dataloader(unrelate,batch_size)
    ster_race_dataloader=get_dataloader(ster_race, ster_batch_size)
    ster_gender_dataloader=get_dataloader(ster_gender, ster_batch_size)
    ster_profession_dataloader=get_dataloader(ster_profession, ster_batch_size)
    ster_religion_dataloader=get_dataloader(ster_religion, ster_batch_size)
    anti_race_dataloader=get_dataloader(anti_race, batch_size)
    anti_gender_dataloader=get_dataloader(anti_gender, batch_size)
    anti_profession_dataloader=get_dataloader(anti_profession, batch_size)
    anti_religion_dataloader=get_dataloader(anti_religion, batch_size)

    return (
        anti_dataloader, 
        ster_dataloader, 
        unrelate_dataloader,  
        ster_race_dataloader,
        ster_gender_dataloader,
        ster_profession_dataloader,
        ster_religion_dataloader,
        anti_race_dataloader,
        anti_gender_dataloader,
        anti_profession_dataloader,
        anti_religion_dataloader
    )






