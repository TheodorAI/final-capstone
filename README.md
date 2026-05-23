# SmartQG: 基于逻辑图谱检索增强生成的多题型自动出题系统

SmartQG 是一个面向计算机科学（CS）课程的自动化高质量出题框架。系统通过构建**可执行算法知识图谱（GEAKG）**，结合**种子-剪枝逻辑图检索**机制，引导大语言模型合成跨概念关系并触发对抗性边界条件，从而生成具有心理测量学严谨性的考题。

## 🚀 核心贡献

1. **可执行算法知识图谱（GEAKG）**：
   - 包含 **4,659 个节点** 和 **37,998 条三元组关系**
   - 编码五种有向过程关系：`HAS_STEP`、`TRUE_BRANCH`、`FALSE_BRANCH`、`PRODUCES_OUTPUT`、`HAS_COMPLEXITY`

2. **种子-剪枝检索与主题自适应**：
   - 动态调整相似度阈值（计算题 0.25 vs 概念题 0.10）和子图扩展上限
   - 通过余弦相似度剪枝消除邻居噪音

3. **多题型生成与自动校正**：
   - 支持 **5 种题型**：单选题、多选题、判断题、填空题、开放问答
   - 格式感知的难度自校正循环，确保大学级别严谨性

## 📂 项目结构

```
├── kg_loader.py              # KG 常驻守护进程 — 将知识图谱加载到内存
├── single_query.py            # 轻量客户端 — 连接 daemon 生成单道题目
├── build_kg.py                # KG 构建脚本 — 从教材抽取三元组 + 构建题库
├── global_knowledge_graph.json # 知识图谱数据（三元组）
├── question_bank.json          # 题库（few-shot 示例）
├── GraphRAG-Bench/
│   └── textbooks/             # 教材原始数据
└── rag_system/                # 核心组件
    ├── generator.py           # 三种生成器：无检索 / 向量 RAG / 图谱 RAG
    ├── retriever.py           # 检索器：向量搜索 + 图扩展 + 混合排序
    ├── knowledge_graph.py     # 知识图谱数据结构
    ├── evaluator.py           # 自动评分器（7 维度）
    └── logger.py              # 日志工具
```

## 🛠️ 安装

```bash
# 1. 安装依赖
pip install openai sentence-transformers faiss-cpu rank_bm25 python-dotenv

# 2. 配置 API Key
echo "DEEPSEEK_API_KEY=your_key_here" > .env

# 3. （可选）放置教材数据到 GraphRAG-Bench/textbooks/
```

## 📖 使用流程

### 第一步：构建知识图谱

```bash
python build_kg.py --mode extract_only
```

从教材中抽取三元组，生成 `global_knowledge_graph.json`。

### 第二步：启动守护进程

```bash
python kg_loader.py
```

将 KG、向量模型、题库一次性加载到内存中，监听 Unix socket。

### 第三步：生成题目

```bash
# 单选题（默认）
python single_query.py --topic "merge sort"

# 指定检索策略与题型
python single_query.py --topic "hash table" --generator vector_rag --format true_false

# 批量查询
printf '{"topic":"merge sort"}\n{"topic":"BFS","generator":"vector_rag"}\n' | \
  python single_query.py
```

## ⚙️ 三种生成策略

| 策略 | 参数 | 说明 |
|---|---|---|
| 无检索 | `--generator no_retrieval` | 纯 LLM 知识，无外部上下文 |
| 向量 RAG | `--generator vector_rag` | 向量检索教材片段 + 生成 |
| 图谱 RAG | `--generator graph_rag`（默认） | 知识图谱子图检索 + 生成 |

## 📝 五种题型

| 题型 | `--format` | 说明 |
|---|---|---|
| 单选题 | `mcq_single`（默认） | 四选一 |
| 多选题 | `mcq_multi` | 多选 |
| 判断题 | `true_false` | 真/假判断 |
| 填空题 | `fill_blank` | 1-2 空 |
| 开放问答 | `open_answer` | 开放性问题 |

## 📊 生成耗时

每次生成会输出耗时分解：

```
题目: ...
答案: ...

生成方式: generated
总耗时:   31.4s
  ├─ 检索: 0.4s
  └─ 生成: 31.0s (含 LLM 调用 + 难度过滤)
```

耗时主要花在 LLM 生成环节（含难度评分与重试循环），检索时间极短。

## 🎓 评分维度

自动评分器从 7 个维度评估题目质量：
**相关性(5%) · 多样性(10%) · 正确性(20%) · 诊断力(20%) · 多跳依赖(15%) · 边界触发(20%) · 图关系深度(10%)**
