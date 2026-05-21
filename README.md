# 面向个人的多模态 RAG 知识库问答系统

一个基于 `FastAPI + Chroma + BM25 + 本地 Embedding + 本地 Reranker + DeepSeek/GLM API` 的多模态 RAG 项目，面向个人知识管理和轻量企业知识库场景。系统支持多格式文档上传、OCR 文字提取、VLM 图像语义补充、混合检索、结构化回答、引用返回，以及按文档范围进行精确查询。

## 项目定位

这个项目不是简单的“文档上传 + 向量检索 + 大模型回答”，而是围绕真实问答链路做了几层增强：

- 离线侧：文档解析、清洗、切块、OCR/VLM 增强、向量化、持久化索引
- 在线侧：查询改写、文档级粗筛、BM25 + 向量混合召回、低置信度二次检索、邻近 chunk 扩展、Rerank 精排、结构化回答
- 工程侧：文档去重、文档替换重建、日志记录、Embedding 缓存、文档级元数据管理

## 技术亮点

### 1. 离线索引构建做了多模态增强

- 支持 `PDF / DOCX / TXT / MD / HTML / PNG / JPG / JPEG`
- 针对复杂文档做了解析与清洗，避免原始抽取文本直接入库
- 对含图文档引入 `OCR + VLM` 双通道：
  - `OCR` 提取可见文字
  - `VLM` 补充截图、页面、图表、界面类内容的视觉语义
- 切块时加入标题附着、噪声过滤、分页后内容清洗，减少低质量 chunk 干扰
- 建立 `BM25 + 向量` 的混合索引，而不是只依赖单一路径召回

### 2. 在线检索链路不是单步检索，而是完整检索编排

- 查询改写：对用户原始问题生成多种检索表达，提升召回覆盖
- 问题路由：对图片相关问题优先提升 `VLM / OCR` 模态片段权重
- 多路召回：向量检索与 BM25 并行召回，再通过 `RRF / weighted` 做融合
- 二次检索：当首轮结果低置信度时，自动触发补充查询与第二轮召回
- 邻近扩展：对高相关 chunk 自动补充相邻片段，降低断章取义
- Rerank 精排：使用本地 Cross-Encoder 做最终排序
- 上下文压缩：只选少量高信号片段拼接 Prompt，控制 token 消耗
- 结构化回答：输出 `answer / conclusion / key_points / citations`

### 3. 为降低 token 成本，增加了文档级检索范围控制

- 查询时支持按 `document_ids` 精确限定文档范围
- 支持按 `title_keyword` 按标题筛选
- 当用户未指定文档、文档数量较多时，系统会先做文档级粗筛，再进入 chunk 级召回
- 前端上线后可以通过“文档列表选择 + 查询”模式，避免全库无差别拼接上下文

### 4. 工程实现上强调可维护性和可替换性

- 基于 `file hash` 做文档级去重，避免重复入库
- 支持上传时通过 `replace_document_id` 替换旧文档并重建索引
- 文档元数据中保留 `title / upload time / filename / hash / chunk_count`
- Embedding 层支持缓存，减少重复向量化成本
- 日志分为应用日志、访问日志、错误日志，便于问题排查
- 文档解析、Embedding、Vector Store、Hybrid Search、LLM 生成分别封装，便于后续替换模型或存储后端

## 当前已实现能力

### 文档处理

- 多格式文档上传
- 文本提取与清洗
- PDF 页面渲染后 OCR
- DOCX 内嵌图片 OCR / VLM 增强
- 图片文档直接 OCR / VLM 处理
- 文档去重
- 文档删除
- 文档替换重建

### 检索增强

- Chroma 持久化向量库
- BM25 稀疏检索
- 向量检索
- RRF / Weighted 融合
- 查询改写
- 低置信度二次检索
- 邻居 chunk 扩展
- 图片类问题的模态优先路由
- 本地 Reranker 精排

### 回答生成

- DeepSeek 兼容接口生成答案
- 图片类问题优先使用 VLM 片段
- 返回结构化字段
- 返回引用来源
- 支持流式输出

### 文档管理

- 文档标题
- 文档上传时间
- 文档列表接口
- 按文档查询
- 全量删除 / 单文档删除

## 技术栈

- Web 框架：`FastAPI`
- 向量数据库：`Chroma`
- 稀疏检索：`BM25`
- Embedding：本地 `BAAI/bge-small-zh-v1.5` 或兼容 OpenAI Embedding API
- Reranker：本地 `cross-encoder/ms-marco-MiniLM-L-12-v2`
- LLM：`DeepSeek` 兼容 `chat/completions`
- VLM：兼容 OpenAI 风格视觉接口，当前已接入 `GLM-4V-Flash`
- OCR：`Tesseract OCR + pytesseract`
- PDF 处理：`PyMuPDF`

## 系统检索链路

```text
用户问题
  -> 查询改写
  -> 文档级粗筛（可选）
  -> 向量召回
  -> BM25 召回
  -> 融合排序（RRF / Weighted）
  -> 低置信度二次检索（可选）
  -> 邻居 Chunk 扩展（可选）
  -> Rerank 精排
  -> 上下文压缩
  -> LLM 结构化回答
```

## 为什么这个项目更适合放在简历里

如果你要把这个项目写进简历，建议强调的不是“我做了一个 RAG”，而是下面这些更有区分度的点：

- 不是基础向量检索，而是做了 `BM25 + 向量 + Rerank` 的混合检索链路
- 不是纯文本 RAG，而是加入了 `OCR + VLM` 的多模态增强
- 不是只管召回，还做了低置信度二次检索、邻居扩展和上下文压缩
- 不是一次性 Demo，而是支持文档去重、替换重建、日志、缓存和文档级范围控制

可参考的项目描述表达：

> 设计并实现一个面向个人与轻量企业场景的多模态 RAG 知识库问答系统。系统基于 FastAPI、Chroma、BM25、本地 Embedding 与 Cross-Encoder Reranker 构建，支持 PDF、DOCX、Markdown、图片等多格式文档上传；针对图文混排文档引入 OCR 与 VLM 双通道增强，补充可见文字与视觉语义；在线检索阶段实现查询改写、文档级粗筛、多路召回、低置信度二次检索、邻近片段扩展与精排，提升复杂问题下的召回稳定性与回答质量；同时结合文档 Hash 去重、文档替换重建、Embedding 缓存和日志体系，增强系统的工程可维护性。

## 当前接口

### 1. 健康检查

`GET /api/v1/health`

### 2. 上传文档

`POST /api/v1/upload`

表单字段：

- `file`: 上传文件
- `title`: 可选，自定义文档标题
- `replace_document_id`: 可选，替换已有文档时使用

### 3. 问答查询

`POST /api/v1/query`

请求示例：

```json
{
  "query": "这个文档说了什么，图片表达了什么",
  "top_k": 6,
  "stream": false,
  "use_rerank": true
}
```

按文档精确查询：

```json
{
  "query": "报销标准是什么",
  "document_ids": ["your-document-id"],
  "top_k": 6
}
```

按标题范围查询：

```json
{
  "query": "合同里违约责任怎么写",
  "title_keyword": "采购合同",
  "top_k": 6
}
```

### 4. 文档列表

`GET /api/v1/documents`

### 5. 删除单个文档

`DELETE /api/v1/documents/{document_id}`

### 6. 清空全部文档

`DELETE /api/v1/documents`

## 环境要求

- Python `3.11`
- 已安装 `Tesseract OCR`
- 本地 Embedding 模型目录
- 本地 Reranker 模型目录
- 可用的 LLM / VLM API Key

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 复制环境变量模板

```powershell
copy .env.example .env
```

敏感密钥建议单独放到 `.secrets.env`，系统会自动同时加载 `.env` 和 `.secrets.env`。

### 3. 配置基础环境

最少需要确认以下配置：

```env
VECTOR_STORE_PROVIDER=chroma
CHROMA_PERSIST_DIR=data/chroma

ENABLE_OCR=true
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

EMBEDDING_MODEL=E:/your-project/models/bge-small-zh-v1.5
RERANK_MODEL=E:/your-project/models/ms-marco-MiniLM-L-12-v2
```

`.secrets.env` 中保存密钥：

```env
LLM_API_KEY=your-llm-key
VLM_API_KEY=your-vlm-key
```

### 4. 启动服务

```powershell
python main.py
```

或：

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 打开 Swagger

```text
http://127.0.0.1:8000/docs
```

## 关键配置项

### OCR

```env
ENABLE_OCR=true
OCR_LANGUAGE=chi_sim+eng
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### VLM

```env
VLM_ENABLED=true
VLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
VLM_MODEL=glm-4v-flash
VLM_TIMEOUT=120
VLM_MAX_IMAGES_PER_DOCUMENT=3
```

### 检索

```env
HYBRID_SEARCH_ENABLED=true
VECTOR_TOP_K=20
BM25_TOP_K=20
FUSION_MODE=rrf
RRF_K=60
QUERY_REWRITE_ENABLED=true
SECOND_PASS_ENABLED=true
NEIGHBOR_EXPAND_ENABLED=true
DOCUMENT_CANDIDATE_TOP_K=5
DOCUMENT_SCOPE_TRIGGER_COUNT=3
```

### Rerank

```env
RERANK_ENABLED=true
RERANK_INITIAL_K=50
RERANK_FINAL_K=10
```

## 目录结构

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
  cache/
logs/
models/
```

## 当前边界

下面这些方向是后续可继续增强的，但当前版本不应写成“已经完成”：

- 基于 `RAGAS` 的系统化评测
- `MRR / Hit@k` 检索评测看板
- 更完整的前端文档管理界面
- 多租户 / 多部门权限隔离
- 更细粒度的增量热更新调度

## Git 建议忽略

不建议提交：

- `.env`
- `.secrets.env`
- `data/`
- `logs/`
- `models/`
- `.idea/`
- `__pycache__/`
