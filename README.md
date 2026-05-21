# Enterprise RAG

基于 `FastAPI + Chroma + BM25 + Local Embedding + Local Reranker + DeepSeek` 的个人/企业知识库问答系统，支持 OCR、多格式文档解析、混合检索、Rerank 和结构化回答输出。

## 功能概览

- 支持 `PDF`、`DOCX`、`TXT`、`MD`、`HTML`、`PNG`、`JPG`、`JPEG` 上传
- 支持文档清洗、分块、向量化、持久化索引
- 支持 `Chroma + BM25` 混合检索
- 支持 `RRF` / `weighted` 两种融合策略
- 支持本地 `Cross-Encoder Reranker`
- 支持 `OCR` 提取图片和扫描文档文本
- 支持 `VLM` 补充图片/图表语义信息
- 支持查询改写、低置信度二次检索、邻居 chunk 扩展
- 支持结构化回答：`answer`、`conclusion`、`key_points`、`citations`
- 支持文件 Hash 去重、结果去重、上下文压缩
- 支持 `Demo Mode`，没有真实模型时也能演示完整链路

## 当前技术栈

- API: `FastAPI`
- 向量库: `Chroma`
- 稀疏检索: `BM25`
- 向量模型: 本地 `bge-small-zh-v1.5` 或 OpenAI-compatible Embedding API
- 重排模型: 本地 `ms-marco-MiniLM-L-12-v2`
- 大模型: `DeepSeek` 或兼容 OpenAI Chat Completions 的接口
- OCR: `Tesseract OCR + pytesseract`
- PDF 页图处理: `PyMuPDF`

## 检索链路

当前实现的检索链路不是单一向量检索，而是：

1. 查询改写
2. 多 query 混合召回
3. 向量检索
4. BM25 检索
5. `RRF` / `weighted` 融合
6. 低置信度二次检索
7. 邻居 chunk 扩展
8. 本地 Reranker 精排
9. 上下文压缩
10. 结构化回答生成

你可以把这个项目归类为：

- `Hybrid Search`
- `Retrieval + Rerank`
- `OCR/VLM Enhanced RAG`
- `Structured RAG Response`

## OCR 与 VLM

### OCR

当 `ENABLE_OCR=true` 时：

- 图片文档会直接走 OCR
- PDF 每页会渲染成图片后做 OCR
- DOCX 中的内嵌图片也会尝试 OCR

当前默认通过下面配置指定 Tesseract：

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### VLM

当 `VLM_ENABLED=true` 且配置了 `VLM_API_KEY` 时：

- 图片文档可补充图表、版面、标题、示意图等语义信息
- PDF 页图可补充视觉语义
- DOCX 内嵌图片可补充视觉语义

VLM 当前走 OpenAI-compatible `chat/completions` 接口。

## Demo Mode

适合以下场景：

- 没有本地 embedding / reranker 模型
- 没有可用的 LLM API Key
- 只想给面试官演示链路

启用方式：

```env
DEMO_MODE=true
```

启用后：

- Embedding 使用确定性伪向量
- Reranker 不调用真实模型
- LLM 返回本地 mock 的结构化结果

## 环境要求

### 必需

- Python `3.11`
- Windows / Linux / macOS

### OCR 必需

- 已安装 `Tesseract OCR`
- Windows 默认路径示例：

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 正式模式建议

- 本地 Embedding 模型目录
- 本地 Reranker 模型目录
- 可用的 `LLM_API_KEY`

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 复制环境变量模板

```powershell
copy .env.example .env
```

敏感密钥建议单独放到 `.secrets.env`，应用会自动同时加载 `.env` 和 `.secrets.env`。

### 3. 修改 `.env`

最少需要确认这些配置：

```env
VECTOR_STORE_PROVIDER=chroma
CHROMA_PERSIST_DIR=data/chroma

ENABLE_OCR=true
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

EMBEDDING_MODEL=E:/lost/models/bge-small-zh-v1.5
RERANK_MODEL=E:/lost/models/ms-marco-MiniLM-L-12-v2

```

然后在 `.secrets.env` 中保存密钥：

```env
LLM_API_KEY=your-api-key
VLM_API_KEY=your-api-key
```

如果暂时不想启用视觉语义补充：

```env
VLM_ENABLED=false
```

如果要启用：

```env
VLM_ENABLED=true
VLM_MODEL=deepseek-vl2
```

### 4. 启动服务

```powershell
python main.py
```

或：

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 打开接口文档

```text
http://127.0.0.1:8000/docs
```

### 6. 健康检查

```text
GET http://127.0.0.1:8000/api/v1/health
```

## 主要配置项

### 基础

```env
APP_HOST=0.0.0.0
APP_PORT=8000
API_PREFIX=/api/v1
```

### 向量库

```env
VECTOR_STORE_PROVIDER=chroma
CHROMA_PERSIST_DIR=data/chroma
COLLECTION_NAME=enterprise_docs
```

### OCR

```env
ENABLE_OCR=true
OCR_LANGUAGE=chi_sim+eng
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### VLM

```env
VLM_ENABLED=false
VLM_API_BASE=https://api.deepseek.com/v1
VLM_API_KEY=
VLM_MODEL=deepseek-vl2
VLM_TIMEOUT=120
VLM_MAX_IMAGES_PER_DOCUMENT=3
```

### Chunk

```env
CHUNK_STRATEGY=semantic
CHUNK_SIZE=800
CHUNK_OVERLAP=150
CHUNK_MIN_SIZE=200
CHUNK_TITLE_BOOST=true
```

### Retrieval

```env
HYBRID_SEARCH_ENABLED=true
VECTOR_TOP_K=20
BM25_TOP_K=20
FUSION_MODE=rrf
RRF_K=60
QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_MAX_QUERIES=3
SECOND_PASS_ENABLED=true
SECOND_PASS_TOP_K=30
NEIGHBOR_EXPAND_ENABLED=true
NEIGHBOR_EXPAND_WINDOW=1
```

### Rerank

```env
RERANK_ENABLED=true
RERANK_INITIAL_K=50
RERANK_FINAL_K=10
RERANK_MODEL=E:/lost/models/ms-marco-MiniLM-L-12-v2
```

## 常用接口

### 1. 健康检查

`GET /api/v1/health`

### 2. 上传文档

`POST /api/v1/upload`

作用：

- 保存文件
- 文本抽取
- OCR / VLM 增强
- 清洗
- 分块
- 向量化
- 写入 Chroma

重复上传相同内容文件时：

- 系统会基于 `SHA256` 做文档级去重
- 自动跳过重复索引

### 3. 查询问答

`POST /api/v1/query`

请求示例：

```json
{
  "query": "这个文档主要讲了什么？",
  "top_k": 10,
  "stream": false,
  "use_rerank": true
}
```

响应中的 `data` 包含：

- `answer`
- `conclusion`
- `key_points`
- `citations`
- `results`
- `prompt`

### 4. 文档列表

`GET /api/v1/documents`

### 5. 删除文档

`DELETE /api/v1/documents/{document_id}`

## 目录说明

```text
app/
  api/
  models/
  services/
  utils/
data/
  chroma/
  uploads/
  tmp/
models/
logs/
```

## 已验证能力

当前版本已在本机验证：

- 应用可正常启动
- `/api/v1/health` 返回正常
- `Chroma` 可初始化
- `OCR` 可执行
- 查询改写服务可正常加载

## Git 建议忽略内容

不建议上传：

- `.env`
- `data/`
- `logs/`
- `models/`
- `.idea/`
- `__pycache__/`

## 简历可写点

- 设计并实现基于 `FastAPI + Chroma + BM25 + Local Embedding + Cross-Encoder Reranker` 的知识库问答系统，支持多格式文档解析、OCR 增强和结构化回答输出。
- 构建 `BM25 + 向量召回 + RRF/Weighted 融合 + Rerank` 的混合检索链路，并加入查询改写、低置信度二次检索和邻居 chunk 扩展以提升召回率与回答稳定性。
- 基于文件 Hash 实现文档级去重，结合 embedding 缓存、上下文压缩和引用映射提升索引效率与回答可追溯性。

## 当前限制

- `VLM` 需要你提供兼容 OpenAI Chat Completions 的视觉模型接口
- `OCR` 依赖本机安装的 `Tesseract OCR`
- 当前评测体系、RAGAS、MRR、Hit@k 还没有实现
