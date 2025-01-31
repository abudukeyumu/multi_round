# 文档检索系统数据准备工具

本项目提供了一系列工具，用于准备和处理文档检索系统所需的数据。

## 使用步骤

### 1. 下载维基百科语料库

首先需要下载维基百科数据：

```bash
python download_resources.py --output_dir=<数据保存目录> --resource_name=wiki_corpus
```

参数说明：
- `output_dir`: 指定数据保存的目录路径
- `resource_name`: 指定要下载的资源名称，此处为wiki_corpus

### 2. 合并数据为TSV格式

将下载的文件合并为单个TSV文件：

```bash
python create_corpus_tsv.py
```

注意：使用前请在脚本中修改相应的文件路径。

### 3. 添加文档索引

在chatrag bench的inscit数据中添加黄金文档的索引：

```bash
python transfer.py
```

注意：使用前请在脚本中修改相应的文件路径。

## 文件说明

- `download_resources.py`: 用于下载维基百科数据的脚本
- `create_corpus_tsv.py`: 用于合并文件为TSV格式的脚本
- `transfer.py`: 用于添加文档索引的脚本

## 注意事项

1. 运行脚本前请确保已安装所需的Python依赖包
2. 请确保有足够的磁盘空间用于存储下载的数据
3. 在运行脚本前，请仔细检查并修改相关文件路径
