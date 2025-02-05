import argparse
import json
import os
import random
import shutil

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class BaseEmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_hard_negatives(self):
        pass

    def apply_query_prompt_template(self, query):
        return query

    def apply_context_prompt_template(self, context):
        return context

    def apply_negative_prompt_template(self, negatives):
        return [
            self.apply_context_prompt_template(negative)
            for negative in negatives
        ]

    def write_jsonl(self, path):
        with open(os.path.join(path), "w") as f:
            for item in self:
                json_line = json.dumps(item)
                f.write(json_line + '\n')
            print(f"All items in this dataset have been written to {path}")

    def __getitem__(self, idx):
        item = self.data[idx]
        query = self.apply_query_prompt_template(item['query'])
        content = self.apply_context_prompt_template(item["content"])
        negatives = self.apply_negative_prompt_template(item["negatives"])
        return {'query': query, 'content': content, 'negatives': negatives}


class ChatQAEmbeddingTrainDataset(BaseEmbeddingDataset):
    def __init__(
        self,
        paths,
        msl_tokens,
        tokenizer,
        chunk_overlap,
        hard_negative_strategy,
        num_hard_negatives,
        ngram_overlap,
    ):
        assert hard_negative_strategy in ["same_document", "diff_document"]

        self.hard_negative_strategy = hard_negative_strategy
        self.num_hard_negatives = num_hard_negatives
        self.msl_tokens = msl_tokens
        self.chunk_overlap = chunk_overlap
        self.paths = paths
        self.ngram_overlap = ngram_overlap
        self.tokenizer = tokenizer
        self.data = self.create_data()

    def load_files(self):
        lines = []
        for path in self.paths:
            with open(path, "r") as f:
                lines += json.load(f)
        return lines

    def create_data(self):
        synconv_list = []
        lines = self.load_files()
        for sample_idx, synconv_sample in tqdm(
            enumerate(lines),
            desc="Reading and Processing data from SyntheticConvQA dataset...",
            total=len(lines),
        ):
            synconv_object = self.generate_retrieval_object(synconv_sample)
            synconv_object["id"] = sample_idx
            synconv_object["chunks"] = self.chunk_doc(
                str_doc=synconv_object["document"],
                tokenizer=self.tokenizer,
                msl_tokens=self.msl_tokens,
                chunk_overlap=self.chunk_overlap,
            )
            max_recall, max_chunk = 0.0, None
            for chunk in synconv_object["chunks"]:
                recall = self.recall_ngrams(
                    synconv_object["answer"], chunk, self.ngram_overlap
                )
                if recall > max_recall:
                    max_recall = recall
                    max_chunk = chunk
            if (
                max_recall != 0
                and len(synconv_object["chunks"]) - 1 - self.num_hard_negatives
                >= 0
            ):
                synconv_object["content"] = max_chunk
                synconv_object["negatives"] = random.sample(
                    [
                        chunk
                        for chunk in synconv_object["chunks"]
                        if chunk != max_chunk
                    ],
                    self.num_hard_negatives,
                )
                synconv_list.append(synconv_object)
        return synconv_list

    def generate_retrieval_object(self, synconv_sample):
        """
        Split the whole object to conversation, document, and answer
        """
        synconv_object = {
            "query": "",
            "document": "",
            "answer": "",
        }
        for line in synconv_sample["messages"]:
            if line["role"] == "user":
                synconv_object['query'] += "User: " + line["content"] + "\n"
            else:
                synconv_object['query'] += "Agent: " + line["content"] + "\n"
        synconv_object["document"] = synconv_sample["document"]
        synconv_object["answer"] = synconv_sample["answers"][0]
        return synconv_object

    def chunk_doc(self, str_doc, tokenizer, msl_tokens, chunk_overlap):
        """
        Create chunks for raw text string document: no headings, no metadata, just a whole string.
        """
        # Calculate average number of characters per token for the given tokenizer
        # if we are using character level splitting method
        token_char_lengths = [
            len(key) for key, value in tokenizer.vocab.items()
        ]
        avg_char_per_token = sum(token_char_lengths) / len(token_char_lengths)
        msl_chars = avg_char_per_token * msl_tokens

        # chunk_overlap can be a ratio of chunk_size or an actual number of chars
        if chunk_overlap < 1.0:
            chunk_overlap = msl_chars * chunk_overlap

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=msl_chars,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = text_splitter.split_text(str_doc)
        return chunks

    def get_ngrams(self, s, n):
        """Generate n-grams from the input string."""
        tokens = s.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [' '.join(ngram) for ngram in ngrams]

    def recall_ngrams(self, reference, candidate, n):
        """Compute the n-gram recall between reference and candidate strings."""
        ref_ngrams = self.get_ngrams(reference, n)
        cand_ngrams = self.get_ngrams(candidate, n)

        ref_ngram_count = len(ref_ngrams)
        matching_ngram_count = sum(
            1 for ngram in cand_ngrams if ngram in ref_ngrams
        )

        recall = (
            matching_ngram_count / ref_ngram_count if ref_ngram_count > 0 else 0
        )
        return recall

def save_dataset(dataset, output_dir, prefix="train"):
    """
    保存数据集为多种格式
    Args:
        dataset: 处理好的数据集
        output_dir: 输出目录
        prefix: 文件前缀（默认为"train"）
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存为PT格式（用于PyTorch加载）
    pt_path = os.path.join(output_dir, f"{prefix}_data.pt")
    torch.save({
        'data': dataset.data,
        'collator': dataset.tokenizer,  # 保存tokenizer以便后续使用
        'config': {
            'msl_tokens': dataset.msl_tokens,
            'chunk_overlap': dataset.chunk_overlap,
            'num_hard_negatives': dataset.num_hard_negatives,
            'ngram_overlap': dataset.ngram_overlap
        }
    }, pt_path)
    print(f"Dataset saved as PT file: {pt_path}")
    
    # 保存为JSONL格式（便于查看和调试）
    jsonl_path = os.path.join(output_dir, f"{prefix}_data.jsonl")
    dataset.write_jsonl(jsonl_path)
    print(f"Dataset saved as JSONL file: {jsonl_path}")
    
    # 保存数据集统计信息
    stats = {
        'num_samples': len(dataset),
        'config': {
            'msl_tokens': dataset.msl_tokens,
            'chunk_overlap': dataset.chunk_overlap,
            'num_hard_negatives': dataset.num_hard_negatives,
            'ngram_overlap': dataset.ngram_overlap
        }
    }
    stats_path = os.path.join(output_dir, f"{prefix}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset stats saved: {stats_path}")

def main(args):
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/mnt/abu/models/bge-large-en-v1.5')
    
    # 处理训练数据
    chatqa_train_jsonl = os.path.join(args.chatqa_training_dir, "synthetic_convqa/train.json")
    chatqa_train_dataset = ChatQAEmbeddingTrainDataset(
        paths=[chatqa_train_jsonl],
        msl_tokens=512,
        tokenizer=tokenizer,
        chunk_overlap=1000,
        hard_negative_strategy="same_document",
        num_hard_negatives=args.num_hard_negatives,
        ngram_overlap=4,
    )
    
    # 保存处理后的数据集
    save_dataset(chatqa_train_dataset, args.output_dir, prefix="train")
    
    print(f"Dataset processing completed. Files saved in {args.output_dir}")
    print(f"Total samples: {len(chatqa_train_dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cerebras DocChat Embedding Dataset Processing")
    parser.add_argument(
        '--chatqa_training_dir',
        type=str,
        default="../Nvidia-ChatQA/ChatQA-Training-Data",
        help='Download path for nvidia/ChatQA-Training-Data dataset',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./processed_datasets/",
        help='Output folder for generated train files',
    )
    parser.add_argument(
        '--num_hard_negatives',
        type=int,
        default=1,
        help='Number of hard negatives used for training',
    )
    args = parser.parse_args()
    main(args)
