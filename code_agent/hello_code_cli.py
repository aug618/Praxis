from __future__ import annotations

import argparse
import os
import re
import logging
import uuid
import json
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

from core.llm import HelloAgentsLLM
from core.exceptions import HelloAgentsException
from core.config import Config, AVAILABLE_MODELS
from code_agent.agentic import CodeAgent
from code_agent.executors.apply_patch_executor import ApplyPatchExecutor, PatchApplyError
from utils.cli_ui import c, hr, PRIMARY, ACCENT, INFO, WARN, ERROR
from utils.observability import log_event


# 匹配 Codex 风格补丁块（宽松，跨行，允许前导空白或代码围栏）
PATCH_RE = re.compile(r"\s*\*\*\* Begin Patch[\s\S]*?\*\*\* End Patch", re.MULTILINE)
# 备用：从 ```patch/```diff 围栏中提取补丁主体
PATCH_FENCE_RE = re.compile(
    r"```(?:patch|diff|text)?\s*(\*\*\* Begin Patch[\s\S]*?\*\*\* End Patch)\s*```",
    re.MULTILINE,
)


def _extract_patch(text: str) -> str | None:
    """
    从 LLM 响应文本中提取补丁块。
    补丁块通常由 *** Begin Patch 和 *** End Patch 包围。
    """
    # 优先匹配代码围栏内的补丁
    m = PATCH_FENCE_RE.search(text)
    if m:
        return m.group(1)
    # 退回普通匹配（允许前导空白）
    m = PATCH_RE.search(text)
    return m.group(0).strip() if m else None


def _normalize_patch(patch_text: str) -> str:
    """
    规范化补丁文本，以宽容处理模型的一些格式错误。
    - 接受 'Delete File:' / 'Update File:' / 'Add File:' (即使缺少前导 '*** ')
    - 保持执行器所需的标准 Codex 风格格式。
    """
    lines = patch_text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("Add File:", "Update File:", "Delete File:")) and not stripped.startswith("*** "):
            out.append("*** " + stripped)
            continue
        out.append(line)
    return "\n".join(out)


def _patch_requires_confirmation(patch_text: str) -> bool:
    """
    判断补丁是否需要用户确认。
    策略：
    - 包含文件删除操作
    - 涉及文件数量过多 (>= 6)
    - 变更行数过多 (>= 400)
    """
    # MVP: Delete File / too many files / too big => confirm
    if "*** Delete File:" in patch_text:
        return True
    file_ops = patch_text.count("*** Add File:") + patch_text.count("*** Update File:") + patch_text.count("*** Delete File:")
    if file_ops >= 6:
        return True
    changed_lines = 0
    for line in patch_text.splitlines():
        if line.startswith("+") or line.startswith("-"):
            changed_lines += 1
    return changed_lines >= 400


def _load_events(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    events = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _summarize_session(events: list[dict]) -> dict:
    stats = {
        "turns": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "llm_calls": 0,
        "llm_errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prompt_tokens_est": 0,
        "completion_tokens_est": 0,
        "duration_ms": None,
        "start_ts": None,
        "end_ts": None,
    }
    for e in events:
        et = e.get("type")
        if et == "session_start":
            stats["start_ts"] = e.get("ts")
        elif et == "session_end":
            stats["end_ts"] = e.get("ts")
            stats["turns"] = e.get("turns", stats["turns"])
        elif et == "tool":
            stats["tool_calls"] += 1
            if not e.get("ok", True):
                stats["tool_errors"] += 1
        elif et == "llm":
            stats["llm_calls"] += 1
            if not e.get("ok", True):
                stats["llm_errors"] += 1
            stats["prompt_tokens"] += e.get("prompt_tokens") or 0
            stats["completion_tokens"] += e.get("completion_tokens") or 0
            stats["prompt_tokens_est"] += e.get("prompt_tokens_est") or 0
            stats["completion_tokens_est"] += e.get("completion_tokens_est") or 0

    if stats["start_ts"] and stats["end_ts"]:
        start_dt = _parse_ts(stats["start_ts"])
        end_dt = _parse_ts(stats["end_ts"])
        if start_dt and end_dt:
            stats["duration_ms"] = int((end_dt - start_dt).total_seconds() * 1000)
    return stats


def main(argv: list[str] | None = None) -> int:
    """
    CLI 入口点。
    初始化 LLM、CodebaseMaintainer 和 PatchExecutor，并进入交互式循环。
    """
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="HelloAgents-style Code Agent CLI (Codex/Claude-like)")
    parser.add_argument("--repo", type=str, default=".", help="Repository root (workspace). Default: .")
    parser.add_argument("--project", type=str, default=None, help="Project name (default: repo folder name)")
    args = parser.parse_args(argv)

    # 2. 初始化环境和 LLM
    repo_root = Path(args.repo).resolve()
    load_dotenv(dotenv_path=repo_root / ".env", override=False)

    project = args.project or repo_root.name
    config = Config.from_env()
    llm = HelloAgentsLLM()  # auto-detect provider from env
    # reduce noisy HTTP client logs in the CLI
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    logging.getLogger("memory").setLevel(logging.WARNING)

    session_id = uuid.uuid4().hex
    os.environ["CODE_AGENT_SESSION_ID"] = session_id
    turns = 0

    def _end_session(reason: str, exit_code: int, error: str | None = None):
        payload = {
            "reason": reason,
            "exit_code": exit_code,
            "turns": turns,
            "project": project,
            "workspace": str(repo_root),
            "model": llm.model,
            "provider": llm.provider,
        }
        if error:
            payload["error"] = error
        log_event("session_end", payload)

    log_event(
        "session_start",
        {
            "project": project,
            "workspace": str(repo_root),
            "model": llm.model,
            "provider": llm.provider,
        },
    )

    print(c(hr("=", 80), INFO))
    print(c("神秘奇奶龙-你的code管家", PRIMARY))
    print(c(f"workspace: {repo_root}", INFO))
    model_type = "多模态" if llm.is_multimodal else "文本"
    print(c(f"当前模型选择: {llm.model} ({model_type})", INFO))
    print(c(f"state: {Path(config.helloagents_dir).as_posix()}", INFO))
    print(c(hr("=", 80), INFO))

    # Optional preflight to surface auth issues early.
    try:
        _ = llm.invoke([{"role": "user", "content": "ping"}], max_tokens=1)
    except HelloAgentsException as e:
        print(c("LLM 预检失败（通常是 API key/base_url/model 配置问题）。", ERROR))
        print(c(f"error: {e}", ERROR))
        print(c("请检查 .env 中的 DEEPSEEK_API_KEY / LLM_* 配置是否正确。", WARN))
        _end_session("preflight_failed", 2, error=str(e))
        return 2

    # 3. 初始化核心组件（ReAct + tools）
    agent = CodeAgent(repo_root=repo_root, llm=llm, config=config)
    patch_executor = ApplyPatchExecutor(repo_root=repo_root)

    # 4. 进入交互循环
    print(c("输入自然语言需求开始；命令：", INFO))
    print(c("  :quit", ACCENT) + c(" 退出", INFO))
    print(c("  :plan <目标>", ACCENT) + c(" 强制生成计划", INFO))
    print(c("  :model", ACCENT) + c(" 查看/切换模型（多模态模型直接识图，文本模型走 OCR）", INFO))
    print(c("  :stats [current|last|<session_id>]", ACCENT) + c(" 查看会话统计", INFO))
    print()
    print(c("@ 引用语法（多个用逗号/顿号分隔）：", INFO))
    print(c("  @file(a.py, b.png)", ACCENT) + c(" 引用文件（支持图片、代码等）", INFO))
    print(c("  @dir(src/, lib/)", ACCENT) + c(" 引用目录（列出结构+关键文件）", INFO))
    print(c("  示例: @file(main.py, image.png) @dir(src/) 请分析这些代码", INFO))
    try:
        while True:
            try:
                user_in = input(c("👤 > ", PRIMARY))
            except (EOFError, KeyboardInterrupt):
                print("\n" + c("电脑没油了，下次再见", INFO))
                _end_session("user_exit", 0)
                return 0

            if user_in is None:
                continue
            user_in = user_in.strip()
            if not user_in:
                print(c("请提供具体指令或问题。", WARN))
                continue
            turns += 1
            if user_in in {":q", ":quit", "quit", "exit"}:
                print(c("bye", INFO))
                _end_session("user_exit", 0)
                return 0
            if user_in.startswith(":stats"):
                arg = user_in[len(":stats"):].strip()
                log_dir = os.getenv("CODE_AGENT_LOG_DIR") or str(Path(".helloagents") / "logs")
                log_path = Path(log_dir) / "events.jsonl"
                events = _load_events(log_path)
                if not events:
                    print(c("暂无日志数据。", WARN))
                    continue

                current_id = os.getenv("CODE_AGENT_SESSION_ID")
                target_id = None
                if arg == "current" or not arg:
                    target_id = current_id
                elif arg == "last":
                    # 取最后一个 session_end 的 session_id
                    for e in reversed(events):
                        if e.get("type") == "session_end":
                            target_id = e.get("session_id")
                            break
                else:
                    target_id = arg

                if not target_id:
                    print(c("未找到目标会话。", WARN))
                    continue

                session_events = [e for e in events if e.get("session_id") == target_id]
                if not session_events:
                    print(c(f"未找到会话: {target_id}", WARN))
                    continue

                stats = _summarize_session(session_events)
                print(c("📊 会话统计", PRIMARY))
                print(c(f"session_id: {target_id}", INFO))
                if stats["start_ts"]:
                    print(c(f"start: {stats['start_ts']}", INFO))
                if stats["end_ts"]:
                    print(c(f"end: {stats['end_ts']}", INFO))
                if stats["duration_ms"] is not None:
                    print(c(f"duration: {stats['duration_ms']} ms", INFO))
                print(c(f"turns: {stats['turns']}", INFO))
                print(c(f"tool_calls: {stats['tool_calls']} (errors: {stats['tool_errors']})", INFO))
                print(c(f"llm_calls: {stats['llm_calls']} (errors: {stats['llm_errors']})", INFO))
                if stats["prompt_tokens"] or stats["completion_tokens"]:
                    print(c(f"tokens: prompt={stats['prompt_tokens']} completion={stats['completion_tokens']}", INFO))
                print(c(f"tokens_est: prompt≈{stats['prompt_tokens_est']} completion≈{stats['completion_tokens_est']}", INFO))
                continue
            if user_in.startswith(":plan"):
                goal = user_in[len(":plan") :].strip() or "请为当前任务生成一个可执行计划"
                response = agent.registry.execute_tool("plan", goal)
                print("\n" + c("🤖 plan", PRIMARY))
                print(response + "\n")
                continue

            # 模型切换命令
            if user_in == ":model":
                model_list = list(AVAILABLE_MODELS.items())
                
                # 显示当前模型和可用模型列表
                model_type = "多模态 📷" if llm.is_multimodal else "文本 📝"
                print(f"\n当前模型: {c(llm.model, PRIMARY)} ({model_type})")
                print(f"\n可用模型:")
                for i, (name, info) in enumerate(model_list, 1):
                    marker = "→ " if name == llm.model else "  "
                    mtype = "多模态" if info["multimodal"] else "文本"
                    print(f"  {marker}[{i}] {c(name, ACCENT)} [{mtype}]")
                
                # 交互式选择
                try:
                    choice = input(c("\n输入数字或模型名切换（回车取消）: ", INFO)).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
                
                if not choice:
                    continue
                
                # 解析选择
                target_model = None
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(model_list):
                        target_model = model_list[idx][0]
                    else:
                        print(c(f"无效序号（范围 1-{len(model_list)}）", ERROR))
                        continue
                elif choice in AVAILABLE_MODELS:
                    target_model = choice
                else:
                    print(c(f"未知模型: {choice}", ERROR))
                    continue
                
                if target_model:
                    old_key = llm.api_key
                    llm.switch_model(target_model)
                    model_type = "多模态 📷" if llm.is_multimodal else "文本 📝"
                    print(c(f"✓ 已切换到: {target_model} ({model_type})", PRIMARY))
                    
                    # 提示 API key 状态
                    if llm.api_key and llm.api_key != old_key:
                        print(c(f"  API Key: 已自动切换 ({llm.api_key[:8]}...)", INFO))
                    elif llm.api_key:
                        print(c(f"  API Key: 使用当前配置 ({llm.api_key[:8]}...)", INFO))
                    else:
                        info = AVAILABLE_MODELS.get(target_model, {})
                        env_names = info.get("api_key_env", [])
                        if env_names:
                            print(c(f"  ⚠️ 未找到 API Key，请配置: {', '.join(env_names)}", WARN))
                    
                    if llm.is_multimodal:
                        print(c("  图片将直接发送给 LLM 进行理解", INFO))
                    else:
                        print(c("  图片将通过 OCR 提取文字后处理", INFO))
                continue

            # 5. 运行一轮对话（ReAct 可能按需调用终端/笔记/记忆）
            # @file/@dir 引用会在 CodeAgent.run_turn 内部解析
            try:
                response = agent.run_turn(user_in)
            except FileNotFoundError as e:
                print(c(f"文件不存在：{e}", ERROR))
                print(c("提示：使用 @file(路径) 引用文件，例如 @file(main.py, image.png) 请分析", WARN))
                continue
            except HelloAgentsException as e:
                print(c(f"LLM 调用失败: {e}", ERROR))
                continue

            # 对于 direct reply（未经过 ReAct 的控制台打印），在 CLI 里补打一份输出
            if getattr(agent, "last_direct_reply", False):
                print(c("🤖 assistant", PRIMARY))
                print(response)
            
            # 7. 提取并应用补丁
            patch_text = _extract_patch(response)
            if not patch_text:
                continue
            patch_text = _normalize_patch(patch_text)
            # Ignore empty patch blocks
            if patch_text.strip() == "*** Begin Patch\n*** End Patch":
                continue

            needs_confirm = _patch_requires_confirmation(patch_text)
            if needs_confirm:
                # If user just answered y/n as the *current* input, treat it as confirmation for this patch.
                if user_in.strip().lower() in {"n", "no"}:
                    print("已取消补丁应用。")
                    continue
                if user_in.strip().lower() not in {"y", "yes"}:
                    print("\n⚠️ 检测到高风险补丁（删除/大规模变更）。是否应用？(y/n)")
                    ans = input("confirm> ").strip().lower()
                    if ans not in {"y", "yes"}:
                        print("已取消补丁应用。")
                        continue

            try:
                res = patch_executor.apply(patch_text)
                print("\n" + c("✅ Patch applied", PRIMARY))
                print(c(f"files: {', '.join(res.files_changed) if res.files_changed else '(none)'}", INFO))
                if res.backups:
                    print(c(f"backups: {len(res.backups)} (in .helloagents/backups/...)", INFO))

                # 记录到 NoteTool（action）
                agent.note_tool.run({
                    "action": "create",
                    "title": "Patch applied",
                    "content": f"User input:\n{user_in}\n\nPatch:\n\n```text\n{patch_text}\n```\n\nFiles:\n"
                    + "\n".join([f"- {p}" for p in res.files_changed]),
                    "note_type": "action",
                    "tags": [project, "patch_applied"],
                })
            except PatchApplyError as e:
                print("\n" + c(f"❌ Patch failed: {e}", ERROR))
                agent.note_tool.run({
                    "action": "create",
                    "title": "Patch failed",
                    "content": f"Error: {e}\n\nUser input:\n{user_in}\n\nPatch:\n\n```text\n{patch_text}\n```\n",
                    "note_type": "blocker",
                    "tags": [project, "patch_failed"],
                })
                continue

    except Exception as e:
        print(c(f"❌ CLI 异常退出: {e}", ERROR))
        _end_session("crash", 1, error=str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
