"""OCR Tool - 图片文字提取工具

当用户使用文本模型时，通过 OCR 提取图片中的文字，注入到上下文中。
支持 MCP 后端调用 OCR 服务。

使用场景：
- 代码截图识别
- 报错截图提取
- 文档图片转文字
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolParameter


class OCRTool(Tool):
    """OCR 工具 - 支持多种后端
    
    后端优先级：
    1. MCP OCR 服务（如果配置了）
    2. 本地 OCR（如 tesseract，如果安装了）
    3. 返回失败提示
    """
    
    def __init__(
        self,
        mcp_server_command: Optional[List[str]] = None,
        mcp_tool_name: str = "ocr",
        fallback_to_local: bool = True,
    ):
        """
        初始化 OCR 工具
        
        Args:
            mcp_server_command: MCP 服务器启动命令，如 ["npx", "ocr-mcp-server"]
            mcp_tool_name: MCP 中 OCR 工具的名称
            fallback_to_local: 如果 MCP 失败，是否尝试本地 OCR
        """
        super().__init__(
            name="ocr",
            description="图片文字提取工具 - 从图片中识别并提取文字内容"
        )
        self.mcp_server_command = mcp_server_command
        self.mcp_tool_name = mcp_tool_name
        self.fallback_to_local = fallback_to_local
        self._mcp_client = None
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="image_path",
                type="string",
                description="图片文件路径",
                required=True,
            ),
        ]
    
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行 OCR"""
        image_path = parameters.get("image_path", "")
        if not image_path:
            return "错误：未提供图片路径"
        
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            return f"错误：图片文件不存在: {path}"
        
        # 1. 尝试 MCP OCR
        if self.mcp_server_command:
            result = self._ocr_via_mcp(path)
            if result and not result.startswith("错误"):
                return result
        
        # 2. 尝试本地 tesseract
        if self.fallback_to_local:
            result = self._ocr_via_tesseract(path)
            if result and not result.startswith("错误"):
                return result
        
        return "OCR 失败：未配置可用的 OCR 后端。请配置 MCP OCR 服务或安装 tesseract。"
    
    def _ocr_via_mcp(self, image_path: Path) -> Optional[str]:
        """通过 MCP 调用 OCR"""
        try:
            # 这里需要根据你的 MCP OCR 服务实现
            # 示例：调用 MCP 工具
            from tools.builtin.mcp_wrapper_tool import MCPWrapperTool
            
            if self._mcp_client is None:
                self._mcp_client = MCPWrapperTool(
                    name="ocr_mcp",
                    server_command=self.mcp_server_command,
                )
            
            # 读取图片并 base64 编码
            image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            
            result = self._mcp_client.run({
                "action": "call_tool",
                "tool_name": self.mcp_tool_name,
                "arguments": {
                    "image": image_base64,
                    "image_path": str(image_path),
                }
            })
            return result
        except Exception as e:
            return f"MCP OCR 错误: {e}"
    
    def _ocr_via_tesseract(self, image_path: Path) -> Optional[str]:
        """通过本地 tesseract 进行 OCR"""
        try:
            # 检查 tesseract 是否安装
            result = subprocess.run(
                ["tesseract", str(image_path), "stdout", "-l", "chi_sim+eng"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                text = result.stdout.strip()
                if text:
                    return text
                return "OCR 结果为空（图片中可能没有可识别的文字）"
            return f"tesseract 错误: {result.stderr}"
        except FileNotFoundError:
            return None  # tesseract 未安装，返回 None 让调用方知道
        except subprocess.TimeoutExpired:
            return "OCR 超时"
        except Exception as e:
            return f"本地 OCR 错误: {e}"


def extract_text_from_image(
    image_path: str | Path,
    mcp_server_command: Optional[List[str]] = None,
) -> str:
    """
    便捷函数：从图片提取文字
    
    Args:
        image_path: 图片路径
        mcp_server_command: MCP OCR 服务命令（可选）
        
    Returns:
        提取的文字，或错误信息
    """
    tool = OCRTool(mcp_server_command=mcp_server_command)
    return tool.run({"image_path": str(image_path)})
