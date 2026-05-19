# Enterprise RAG

基于 FastAPI、Chroma、本地 Embedding、本地 Reranker 和 DeepSeek 的企业级 RAG 检索增强生成系统。

## 项目亮点

- 将分散在 PDF、Word、Markdown、网页中的知识整理为可检索、可问答的统一知识库
- 支持 `PDF`、`DOCX`、`TXT`、`MD`、`HTML` 文档上传、解析、清洗和切块
- 实现 `Chroma + BM25` 混合检索，并支持 `RRF` / `weighted` 融合
- 支持本地 `Reranker` 精排，提升检索结果相关性
- 支持结构化问答输出：`answer`、`conclusion`、`key_points`、`citations`
- 支持文件哈希去重、检索结果去重、上下文压缩
- 支持 `Demo 模式`，不依赖真实大模型 API 和本地模型也可完成系统演示

## 技术亮点

- 独立实现企业级 RAG 知识库问答系统，技术栈涵盖 `FastAPI + Chroma + BM25 + Local Embedding + Local Reranker + DeepSeek`
- 构建混合检索链路，将向量检索与 BM25 词法检索结合，提升召回覆盖率与检索稳定性
- 实现本地化 Embedding 与本地化 Reranker，支持离线部署，降低外部模型依赖
- 构建文档处理流水线，支持多格式解析、文本清洗、智能切块和索引入库
- 实现文档去重、结果去重、上下文压缩和结构化回答输出，增强可追溯性与前端集成能力

## 检索策略

当前项目实现的是混合检索，不是单一向量检索：

- 向量检索：基于 Embedding 相似度在 Chroma 中召回
- BM25 检索：基于关键词匹配做词法召回
- 融合方式：
  - 默认 `RRF`
  - 可切换 `weighted`
- 融合后再经过本地 `Reranker` 精排

你的项目可归类为：

- `Hybrid Search`
- `Retrieval + Rerank`
- `Structured RAG Response`

## 返回结构

`POST /api/v1/query` 返回的 `data` 中包含：

- `answer`：兼容文本展示
- `conclusion`：结构化结论
- `key_points`：结构化要点
- `citations`：引用片段来源
- `results`：检索结果原始数据
- `prompt`：实际发送给模型的上下文提示词

## Demo 模式

### 作用

`Demo 模式` 用于解决以下问题：

1. HR 或面试官没有本地模型
2. HR 或面试官没有真实 `DeepSeek API Key`

开启后，系统仍然可以：

- 启动接口服务
- 上传文档
- 文档切块
- 建立演示索引
- 执行检索
- 返回结构化问答结果

但不会调用真实大模型，也不会依赖真实本地 Embedding / Reranker 模型。

### 如何开启

在 `.env` 中设置：

```env
DEMO_MODE=true
```

### Demo 模式行为

- Embedding：使用确定性伪向量，保证流程可运行
- Reranker：直接保留当前排序结果
- LLM：返回本地 mock 的结构化答案

### 是否影响正式模式

不会。

当：

```env
DEMO_MODE=false
```

系统会恢复为正式模式：

- 使用本地 Embedding 模型
- 使用本地 Reranker 模型
- 调用真实 DeepSeek API

所以 `Demo 模式` 只是一个开关，不会破坏原有正式能力。

## 仓库中不上传的内容

以下内容不建议上传到 GitHub：

- `.env`
- `data/`
- `logs/`
- `models/`
- `.idea/`
- `__pycache__/`

原因：

- `.env` 可能包含真实密钥
- `data/`、`logs/`、`models/` 属于本地运行数据或大文件
- `models/` 体积较大，不适合普通 Git 仓库

## 别人拿到仓库后能否运行

可以，但分两种情况。

### 1. 只想演示系统跑起来

推荐直接使用 `Demo 模式`。

只需要：

1. 安装 Python 依赖
2. 复制 `.env.example` 为 `.env`
3. 设置 `DEMO_MODE=true`
4. 启动服务

这种方式不需要：

- 本地 Embedding 模型
- 本地 Reranker 模型
- DeepSeek API Key

### 2. 想运行完整正式能力

需要额外准备：

1. 本地 Embedding 模型目录
2. 本地 Reranker 模型目录
3. `LLM_API_KEY`

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 复制环境变量模板

```powershell
copy .env.example .env
```

### 3. 选择运行模式

#### 方案 A：演示模式

修改 `.env`：

```env
DEMO_MODE=true
```

#### 方案 B：正式模式

修改 `.env`：

```env
DEMO_MODE=false
LLM_API_KEY=your-deepseek-key
EMBEDDING_MODEL=E:/your-project/models/bge-small-zh-v1.5
RERANK_MODEL=E:/your-project/models/ms-marco-MiniLM-L-12-v2
```

### 4. 启动服务

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

或：

```powershell
python main.py
```

### 5. 打开接口文档

```text
http://127.0.0.1:8000/docs
```

## 常用接口

### 健康检查

`GET /api/v1/health`

### 上传文档

`POST /api/v1/upload`

作用：

- 保存文件
- 解析文本
- 清洗
- 切块
- 向量化
- 写入 Chroma

重复上传同一文件时：

- 系统会基于 `SHA256` 识别重复文档
- 自动跳过重复索引

### 查询问答

`POST /api/v1/query`

请求示例：

```json
{
  "query": "这个文档说了什么",
  "top_k": 10,
  "stream": false,
  "use_rerank": true
}
```

响应示例：

```json
{
  "success": true,
  "message": "ok",
  "data": {
    "answer": "结论：...\n\n关键要点：\n- ...\n- ...\n\n依据：[1][2]",
    "conclusion": "这份文档主要介绍了某主题的核心内容。",
    "key_points": [
      "介绍了核心定义",
      "介绍了流程",
      "介绍了应用场景"
    ],
    "citations": [
      {
        "ref": "[1]",
        "filename": "text.docx"
      }
    ],
    "query": "这个文档说了什么",
    "results": [],
    "prompt": "..."
  }
}
```

### 文档列表

`GET /api/v1/documents`

### 删除文档

`DELETE /api/v1/documents/{document_id}`

## 发布到 GitHub 的建议

建议保留：

- `app/`
- `main.py`
- `.env.example`
- `README.md`
- `requirements.txt`
- `docker-compose.yml`
- `Dockerfile`

建议忽略：

- `.env`
- `data/`
- `logs/`
- `models/`
- `.idea/`
- `__pycache__/`

## 对企业的价值

### 1. 提高知识检索效率

- 把分散文档变成统一知识库
- 降低人工翻文档时间
- 提高内部答疑响应速度

### 2. 降低大模型调用成本

- 通过检索先缩小上下文范围
- 通过上下文压缩减少无效文本
- 有助于减少 token 消耗

### 3. 降低幻觉风险

- 回答基于检索片段
- 输出支持引用来源
- 比纯大模型裸问答更可控

## 简历可写点

- 设计并实现企业级 RAG 知识库问答系统，支持多格式文档解析、混合检索、重排和结构化答案输出
- 基于 `Chroma + BM25 + RRF + Reranker` 构建混合召回与精排链路，提升检索相关性
- 实现文档去重、上下文压缩和结构化引用，降低重复索引、提升回答可追溯性
- 设计 `Demo 模式`，降低项目演示门槛，提升非技术评审场景下的可交付性
