import os
import argparse
import logging
import shutil
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

def setup_logging(output_dir):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

class DragonRetriever(torch.nn.Module):
    """Dragon检索模型"""
    def __init__(self, model_path, device):
        super().__init__()
        # 加载tokenizer和模型
        # self.tokenizer = AutoTokenizer.from_pretrained(f"{model_path}-query-encoder")
        # self.query_encoder = AutoModel.from_pretrained(f"{model_path}-query-encoder").to(device)
        # self.context_encoder = AutoModel.from_pretrained(f"{model_path}-context-encoder").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")
        self.query_encoder = AutoModel.from_pretrained(f"{model_path}").to(device)
        self.context_encoder = AutoModel.from_pretrained(f"{model_path}").to(device)
        self.temperature = 1.0
        
    def forward(self, questions_dict, contexts_dict):
        # 编码问题
        q_outputs = self.query_encoder(
            input_ids=questions_dict['questions_input_ids'],
            attention_mask=questions_dict['questions_attention_mask'],
            token_type_ids=questions_dict['questions_token_type_ids']
        )
        q_embeddings = q_outputs.last_hidden_state[:, 0, :]

        # 编码文档（包括正例和负例）
        ctx_outputs = self.context_encoder(
            input_ids=contexts_dict['ctx_input_ids'].view(-1, contexts_dict['ctx_input_ids'].size(-1)),
            attention_mask=contexts_dict['ctx_attention_mask'].view(-1, contexts_dict['ctx_attention_mask'].size(-1)),
            token_type_ids=contexts_dict['ctx_token_type_ids'].view(-1, contexts_dict['ctx_token_type_ids'].size(-1))
        )
        ctx_embeddings = ctx_outputs.last_hidden_state[:, 0, :]
        
        batch_size = questions_dict['questions_input_ids'].size(0)
        num_contexts = contexts_dict['ctx_input_ids'].size(1)
        ctx_embeddings = ctx_embeddings.view(batch_size, num_contexts, -1)
        
        q_embeddings = q_embeddings.unsqueeze(1)
        scores = torch.bmm(q_embeddings, ctx_embeddings.transpose(1, 2)).squeeze(1)
        scores = scores / self.temperature
        
        return scores
    
    def save_model(self, output_dir):
        """保存模型和tokenizer"""
        # 保存query encoder和tokenizer
        query_encoder_path = os.path.join(output_dir, "query_encoder")
        self.query_encoder.save_pretrained(query_encoder_path)
        self.tokenizer.save_pretrained(query_encoder_path)
        
        # 保存context encoder和tokenizer
        context_encoder_path = os.path.join(output_dir, "context_encoder")
        self.context_encoder.save_pretrained(context_encoder_path)
        self.tokenizer.save_pretrained(context_encoder_path)

class EmbeddingCollator:
    """数据整理器"""
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        questions = [s['query'] for s in batch]
        contexts = [s['content'] for s in batch]
        negatives = [s['negatives'] for s in batch]

        questions_info = self.tokenizer(
            questions,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length,
        )

        ctx_input_ids, ctx_attention_mask, ctx_token_type_ids = [], [], []
        for i in range(len(contexts)):
            sample_contexts = [contexts[i]] + negatives[i]
            context_info = self.tokenizer(
                sample_contexts,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length,
            )

            ctx_input_ids.append(context_info.input_ids.unsqueeze(0))
            ctx_attention_mask.append(context_info.attention_mask.unsqueeze(0))
            ctx_token_type_ids.append(context_info.token_type_ids.unsqueeze(0))

        output = {
            "questions_input_ids": questions_info.input_ids,
            "questions_attention_mask": questions_info.attention_mask,
            "questions_token_type_ids": questions_info.token_type_ids,
            "ctx_input_ids": torch.concatenate(ctx_input_ids),
            "ctx_attention_mask": torch.concatenate(ctx_attention_mask),
            "ctx_token_type_ids": torch.concatenate(ctx_token_type_ids),
        }
        return output

class AverageMeter:
    """用于跟踪指标的类"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pt'):
    """保存检查点"""
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(output_dir, 'model_best.pt')
        shutil.copyfile(checkpoint_path, best_path)

def train_epoch(epoch, model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for i, batch in enumerate(train_loader):
        # 将数据移到设备上
        questions_dict = {k: v.to(device) for k, v in batch.items() if k.startswith('questions_')}
        contexts_dict = {k: v.to(device) for k, v in batch.items() if k.startswith('ctx_')}
        
        # 前向传播
        scores = model(questions_dict, contexts_dict)
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(scores, labels)
        
        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪和优化器步进
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 更新统计
        losses.update(loss.item() * gradient_accumulation_steps)  # 还原实际loss值
        train_loader.set_postfix({'loss': losses.avg})
    
    return losses.avg

def train_dragon(
    train_data_path: str,
    model_path: str,
    output_dir: str,
    num_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
    max_length: int = 512,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 450,
    device: str = "cuda"
):
    """主训练函数"""
    # 设置日志
    setup_logging(output_dir)
    
    # 加载数据
    train_data = torch.load(train_data_path)
    dataset = train_data['data']
    
    # 初始化模型
    model = DragonRetriever(model_path, device)
    # import pdb
    # pdb.set_trace()
    # 创建数据加载器
    collator = EmbeddingCollator(tokenizer=model.tokenizer, max_length=max_length)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True
    )
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练一个epoch
        epoch_loss = train_epoch(
            epoch, model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps
        )
        
        # 保存检查点
        is_best = epoch_loss < best_loss
        best_loss = min(epoch_loss, best_loss)
        
        # 保存模型和检查点
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss
        }
        save_checkpoint(
            checkpoint,
            is_best,
            output_dir,
            filename=f'checkpoint-epoch-{epoch}.pt'
        )
        
        # 保存训练后的模型
        model_save_path = os.path.join(output_dir, f'epoch-{epoch}')
        os.makedirs(model_save_path, exist_ok=True)
        model.save_model(model_save_path)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}')

def main():
    parser = argparse.ArgumentParser(description='Dragon模型训练')
    parser.add_argument('--train_data', type=str, required=True,
                      help='训练数据路径')
    parser.add_argument('--model_path', type=str, required=True,
                      help='预训练模型路径')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出目录')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=450)
    parser.add_argument('--device', type=str, default="cuda")
    
    args = parser.parse_args()
    
    train_dragon(
        train_data_path=args.train_data,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        device=args.device
    )

if __name__ == "__main__":
    main()