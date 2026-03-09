<div align="center">

# Praxis

**一个面向本地代码仓库的智能 AI 编程助手**

[![Python](https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?style=flat-square)](LICENSE)
[![uv](https://img.shields.io/badge/package%20manager-uv-7c3aed?style=flat-square)](https://github.com/astral-sh/uv)
[![hello-agents](https://img.shields.io/badge/powered%20by-hello--agents-f59e0b?style=flat-square)](https://github.com/hello-agents)

[特性](#-核心特性) · [快速开始](#-快速开始) · [架构](#-系统架构) · [配置](#-配置参考) · [使用指南](#-使用指南) · [路线图](#-路线图)

</div>

---

Praxis 是一个基于 [hello-agents](https://github.com/hello-agents) 框架构建的交互式 AI 编程助手，提供类似 Claude Code / Codex 的本地代码库操作体验。Agent 以 **ReAct 范式**驱动，先通过工具收集证据，再生成补丁落盘，全程支持人工干预与安全回滚。

```bash
# 30 秒上手
git clone <repo-url> && cd praxis
uv sync
cp test/.env.example .env  # 填写 API Key
uv run python -m code_agent.hello_code_cli --repo .
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **ReAct 推理引擎** | 思考 → 行动 → 观察 循环，支持多步骤复杂任务 |
| **安全补丁系统** | 原子写入 + 自动备份（`.backup`），危险操作二次确认 |
| **GSSC 上下文流水线** | Gather → Select → Structure → Compress，按需裁剪 Token |
| **四层记忆系统** | WorkingMemory / EpisodicMemory / SemanticMemory / PerceptualMemory |
| **可扩展工具注册表** | 统一的 `ToolRegistry`，内置 6 类工具，支持自定义挂载 |
| **多 LLM 后端** | 兼容 OpenAI / DeepSeek / Qwen 等任意 OpenAI 协议接口 |

---

## 快速开始

### 环境要求

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv)（推荐）或 pip

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd praxis

# 创建虚拟环境并安装依赖（uv 会自动锁版本）
uv venv && uv sync
```

### 配置

在项目根目录创建 `.env`（可参考 `test/.env.example`）：

```dotenv
# ── LLM（必填）──────────────────────────────────────────────
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# ── 可选调优 ────────────────────────────────────────────────
CODE_AGENT_MAX_STEPS=15          # 最大推理步数
HELLOAGENTS_DIR=.helloagents     # 数据目录（notes / memory / sessions）
```

### 启动

```bash
# 分析当前目录下的代码库
uv run python -m code_agent.hello_code_cli --repo .

# 指定任意路径
uv run python -m code_agent.hello_code_cli --repo /path/to/project
```

---

## 使用指南

### 交互示例

```
📂 Repo: /path/to/project  |  Model: deepseek-chat

> 帮我找出 memory/manager.py 中所有公开方法，并写一份简洁的注释

Thought: 先读取文件内容，再逐一分析公开方法
Action : context_fetch[path=memory/manager.py]
Obs    : [文件内容 ...]

Thought: 共发现 5 个公开方法，开始生成注释补丁

*** Begin Patch
Update File: memory/manager.py
...
*** End Patch

Apply patch? [y/N] y
✓ Patch applied (backup: memory/manager.py.backup)
```

### 内置工具速查

| 工具 | 触发关键词示例 | 说明 |
|------|---------------|------|
| `TerminalTool` | `ls`, `rg`, `cat`, `grep` | 安全终端，白名单命令，危险操作需确认 |
| `ContextFetchTool` | 读取文件 / 目录 | 按需获取代码内容，避免全库扫描 |
| `NoteTool` | 记录决策 / 阻塞 | 写入 `.helloagents/notes/` |
| `TodoTool` | 拆解任务 / 追踪进度 | `pending → in_progress → completed` |
| `PlanTool` | `:plan <目标>` | 生成分步执行计划 |
| `MemoryTool` | 长期记忆存取 | SQLite 持久化，存储于 `.helloagents/memory/` |

### 补丁格式

模型输出以下格式时，CLI 将自动检测并提示应用：

```
*** Begin Patch
Update File: src/foo.py
<<<
旧代码
===
新代码
>>>
*** End Patch
```

> **安全机制**：`Delete File`、`git reset --hard`、大规模变更（>50 行）均需用户输入 `y` 确认。

---

## 系统架构

```
┌──────────────────────────────────┐
│         CLI  /  TUI              │  hello_code_cli.py · hello_code_tui.py
└────────────────┬─────────────────┘
                 │
┌────────────────▼─────────────────┐
│           Agent 层               │  ReActAgent · PlanSolveAgent
│                                  │  ReflectionAgent · SimpleAgent
└────────────────┬─────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌───────────────┐  ┌──────────────────┐
│   Core 层     │  │   能力层          │
│  LLM · Msg   │  │ ContextBuilder    │
│  Config · Exc │  │ MemoryManager    │
└───────────────┘  └──────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌───────────────┐  ┌──────────────────┐
│   Tools 层    │  │   Executors 层    │
│ ToolRegistry  │  │ ApplyPatch       │
│ 6 × Builtin   │  │ Executor         │
└───────────────┘  └──────────────────┘
```

### 模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| **Core** | `core/` | LLM 统一接口、消息抽象、配置、异常体系 |
| **Agents** | `agents/` | ReAct / Plan / Reflection / Simple 四种范式 |
| **Code Agent** | `code_agent/` | CLI/TUI 入口、CodeAgent 主循环、Prompt 模板 |
| **Context** | `context/` | GSSC 流水线：按 Token 预算聚合多源上下文 |
| **Memory** | `memory/` | 四层记忆 + RAG 流水线 + Qdrant/Neo4j 存储后端 |
| **Tools** | `tools/` | 工具基类、注册表、工具链、异步执行器 |
| **Utils** | `utils/` | UI 渲染、日志、序列化、会话管理、补丁工具 |

---

## 项目结构

```
codeGamer/
├── agents/                   # Agent 范式实现
│   ├── react_agent.py
│   ├── plan_solve_agent.py
│   ├── reflection_agent.py
│   └── simple_agent.py
├── code_agent/               # 主应用
│   ├── hello_code_cli.py     # CLI 入口
│   ├── hello_code_tui.py     # TUI 入口（Textual）
│   ├── agentic/
│   │   └── code_agent.py     # CodeAgent 主循环
│   ├── executors/
│   │   └── apply_patch_executor.py
│   └── prompts/              # 系统提示词模板
├── core/                     # 核心抽象层
├── context/                  # GSSC 上下文构建
├── memory/                   # 多层记忆 + RAG
│   ├── types/                # Working / Episodic / Semantic / Perceptual
│   ├── storage/              # Document / Qdrant / Neo4j
│   └── rag/                  # 文档处理 + 检索流水线
├── tools/                    # 工具系统
│   └── builtin/              # 内置工具集
├── utils/                    # 公共工具函数
├── test/                     # 测试 & 配置示例
│   └── .env.example
└── pyproject.toml
```

---

## 配置参考

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_BASE_URL` | `https://api.openai.com` | OpenAI 兼容接口地址 |
| `LLM_MODEL` | `gpt-4o` | 模型名称 |
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key（与 `OPENAI_API_KEY` 二选一） |
| `CODE_AGENT_MAX_STEPS` | `15` | 单次任务最大推理轮数 |
| `HELLOAGENTS_DIR` | `.helloagents` | 持久化数据根目录 |

---

## 路线图

- [ ] **会话恢复**：断点续传，自动加载上次摘要
- [ ] **原子化终端工具**：将 `TerminalTool` 拆分为 `ReadFileTool`、`SearchTool` 等粒度更细的工具
- [ ] **Note Tool 重构**：结构化标签 + 全文检索
- [ ] **记忆系统升级**：向量召回 + 图谱关联，提升长期记忆精度
- [ ] **MCP 协议支持**：接入 Model Context Protocol 标准工具链

---

## 贡献

欢迎提交 Issue 和 Pull Request。在开始之前，请阅读以下约定：

1. 分支命名：`feat/<功能名>`、`fix/<问题简述>`
2. 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)
3. 新特性请附带对应的测试用例

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

<div align="center">

如果这个项目对你有帮助，欢迎点个 ⭐

</div>
