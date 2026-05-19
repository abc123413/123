# Enterprise RAG

基于 FastAPI、Chroma、本地 Embedding、本地 Reranker 和 DeepSeek 的企业级 RAG 检索增强生成系统。

## 项目优点
 把分散在 PDF、Word、Markdown、网页中的知识转成可检索、可问答的统一知识库
  - 降低员工查文档、翻资料、问同事的时间成本
  - 缩短新人培训和知识传递周期
  - 支持客服、售前、交付、内部运营等场景快速获取标准答案
  - 提升回答一致性，减少人工口径不统一的问题
  - 通过引用和结构化输出增强可追溯性，方便合规和审计

 ## 技术亮点
  独立实现企业级 RAG 知识库问答系统，基于 FastAPI + Chroma + BM25 + 本地 Embedding + 本地 Reranker + DeepSeek
  - 设计并落地混合检索链路，融合向量检索与 BM25 关键词检索，提升召回覆盖率与检索稳定性
  - 实现本地化 Embedding 与本地化 Reranker，支持离线部署，降低外部模型依赖与在线服务风险
  - 构建文档处理流水线，支持 PDF / DOCX / TXT / MD / HTML 多格式解析、文本清洗、智能切块和索引入库
  - 实现文档去重、结果去重、上下文压缩和结构化回答输出，提升检索质量、可追溯性与前端集成效率
  - 设计结构化问答响应，输出 conclusion / key_points / citations，增强知识库产品化能力与可审计性



## 功能概览

- 支持上传 `PDF`、`DOCX`、`TXT`、`MD`、`HTML`
- 文档清洗、智能切块、向量缓存
- 本地 Embedding，支持离线运行
- Chroma 向量检索 + BM25 混合检索
- 本地 Reranker 精排
- DeepSeek 生成回答
- 查询响应支持结构化字段：
  - `answer`
  - `conclusion`
  - `key_points`
  - `citations`
- 支持文档去重
  - 基于文件 `SHA256`
- 支持上下文压缩
  - 过滤目录型噪声
  - 优先保留高信息密度片段

## 项目结构

```text
project/
├── app/
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── main.py
└── README.md
```



## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 复制环境变量模板

```powershell
copy .env.example .env
```

### 3. 填写最少必要配置

必须填写：

- `LLM_API_KEY`

推荐本地模型配置：

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=E:/your-project/models/bge-small-zh-v1.5
EMBEDDING_DIMENSION=512
EMBEDDING_DEVICE=cpu
EMBEDDING_NORMALIZE=true

RERANK_ENABLED=true
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

## 模型说明

本仓库默认不上传 `models/` 目录。

你需要自行下载以下模型到本地：

### Embedding 模型

推荐目录：

```text
models/bge-small-zh-v1.5
```

推荐模型：

```text
BAAI/bge-small-zh-v1.5
```

### Reranker 模型

推荐目录：

```text
models/ms-marco-MiniLM-L-12-v2
```

推荐模型：

```text
cross-encoder/ms-marco-MiniLM-L-12-v2
```

建议方式：

- 提前下载模型到本地
- 在 `.env` 中填写本地绝对路径
- 运行时不依赖 Hugging Face 在线下载

## API

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
  "top_k": 5,
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
    "conclusion": "这份文档主要介绍了 RAG 的定义、流程、优势和应用场景。",
    "key_points": [
      "介绍了 RAG 的核心定义。",
      "介绍了标准流程。",
      "介绍了五大核心组件。"
    ],
    "citations": [
      {
        "ref": "[1]",
        "filename": "text.docx"
      },
      {
        "ref": "[2]",
        "filename": "text.docx"
      }
    ],
    "query": "这个文档说了什么",
    "results": [],
    "prompt": "..."
  }
}
```

结构化字段说明：

- `answer`：兼容文本答案
- `conclusion`：结构化结论
- `key_points`：结构化要点
- `citations`：结构化引用
- `results`：检索片段原始数据
- `prompt`：实际送给模型的上下文提示词

### 文档列表

`GET /api/v1/documents`

### 删除文档

`DELETE /api/v1/documents/{document_id}`

## 检索与生成策略

当前实现包含：

- 向量检索
- BM25 检索
- `RRF` / 加权融合
- Reranker 精排
- 上下文压缩
- 固定回答模板
- 结构化响应解析

## 当前工程增强

- 文档去重
- 历史数据兼容
- metadata 空值容错
- chunk 边界优化
- 上下文噪声过滤
- 结构化答案字段


## 注意事项

- `.env.example` 仅作模板，不包含真实密钥
- 修改切块策略后，旧文档不会自动重建索引
  - 需要删除后重新上传
- 首次本地模型加载会稍慢
- 当前文档元数据持久化在：
  - `data/documents.json`
