""" LLM 模型设置

统一入口：修改下方 llm = ... 即可切换大模型，所有 agents 自动使用。
API 密钥从 `.env` 读取：优先 `test1.4.1-AndroidWorld/.env`，其次 `GUI_agent/.env`。
"""
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env：优先本目录（test1.4.1-AndroidWorld/.env），其次上层 GUI_agent/.env（兼容旧路径）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _env in (_PROJECT_ROOT / ".env", _PROJECT_ROOT.parent / ".env"):
    if _env.is_file():
        load_dotenv(_env)

# 不走系统环境变量里的 HTTP(S)_PROXY，避免本地代理未开或返回 403 时 LLM 请求静默失败。
_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=30.0)
_HTTP_SYNC = httpx.Client(trust_env=False, timeout=_HTTP_TIMEOUT)
_HTTP_ASYNC = httpx.AsyncClient(trust_env=False, timeout=_HTTP_TIMEOUT)


def _build_chat_model(*, model: str, base_url: str, api_key: str) -> ChatOpenAI:
    """统一创建 ChatOpenAI，并关闭 thinking，避免 tool-calling 协议冲突。
    
    注意：部分模型（如 Moonshot、ECNU）对 temperature 有限制，
    这里不设置 temperature，使用模型默认值。
    """
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        http_client=_HTTP_SYNC,
        http_async_client=_HTTP_ASYNC,
        # 兼容不同服务商字段，尽量确保关闭 thinking
        extra_body={"enable_thinking": False, "thinking": {"type": "disabled"}},
    )


def _build_siliconflow_vl_chat_model(
    *,
    model: str,
    api_key: str,
    temperature: float = 0.5,
    top_p: float = 0.5,
) -> ChatOpenAI:
    """SiliconFlow 视觉模型专用：不传 enable_thinking / thinking 等扩展字段。

    Qwen3-VL-8B-Instruct 等模型在 SiliconFlow 上若携带 enable_thinking（即便为 False）
    会直接 400，因此与 _build_chat_model 分离，避免影响其他服务商模型。
    """
    base_model = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        http_client=_HTTP_SYNC,
        http_async_client=_HTTP_ASYNC,
    )
    return base_model.bind(temperature=temperature, top_p=top_p)

class LLM_ecnu_plus:
    def __init__(self):
        self.model = _build_chat_model(
            model="ecnu-plus",
            api_key=os.getenv("ECNU_API_KEY", ""),
            base_url="https://chat.ecnu.edu.cn/open/api/v1",
        )

class LLM_ecnu_max:
    def __init__(self):
        self.model = _build_chat_model(
            model="ecnu-max",
            api_key=os.getenv("ECNU_API_KEY", ""),
            base_url="https://chat.ecnu.edu.cn/open/api/v1",
        )

class LLM_deepseek:
    def __init__(self):
        self.model = _build_chat_model(
            model="deepseek-v4-pro",
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )

class LLM_siliconflow_deepseek:
    def __init__(self):
        self.model = _build_chat_model(
            model="deepseek-ai/DeepSeek-V4-Pro",
            api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            base_url="https://api.siliconflow.cn/v1",
        )


class LLM_moonshot_kimi:
    def __init__(self):
        # Moonshot 的 kimi-k2.6 模型只支持 temperature=1
        self.model = ChatOpenAI(
            model="kimi-k2.6",
            base_url="https://api.moonshot.cn/v1",
            api_key=os.getenv("MOONSHOT_API_KEY", ""),
            http_client=_HTTP_SYNC,
            http_async_client=_HTTP_ASYNC,
            temperature=1,
            extra_body={"enable_thinking": False},
        )

class LLM_QWEN3_5_9B:
    def __init__(self):
        self.model = _build_chat_model(
            model="Qwen/Qwen3.5-9B",
            api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            base_url="https://api.siliconflow.cn/v1",
        )

class LLM_QWEN3_VL_8B_INSTRUCT:
    def __init__(self, temperature: float = 0.0):
        self.model = _build_siliconflow_vl_chat_model(
            model="Qwen/Qwen3-VL-8B-Instruct",
            api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            temperature=temperature,
        )

class LLM_siliconflow_vision:
    def __init__(self):
        self.model = _build_chat_model(
            model="zai-org/GLM-4.6V",
            api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            base_url="https://api.siliconflow.cn/v1",
        )

class LLM_siliconflow_minimax:
    def __init__(self):
        self.model = _build_chat_model(
            model="Pro/MiniMaxAI/MiniMax-M2.5",
            api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            base_url="https://api.siliconflow.cn/v1",
        )

class LLM_QWEN_VL_PLUS:
    def __init__(self, temperature: float = 0.0):
        # 先获取基础模型实例（不传 temperature）
        base_model = _build_chat_model(
            model="qwen-vl-plus",
            api_key=os.getenv("QWEN_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # ✅ 通过 bind 注入 temperature 等调用参数
        self.model = base_model.bind(
            temperature=temperature,
            # 可同时绑定其他参数
            top_p=1.0,
            # max_tokens=2048,
        )

class LLM_QWEN3_5_FLASH:
    def __init__(self, temperature: float = 0.0):
        # 先获取基础模型实例（不传 temperature）
        base_model = _build_chat_model(
            model="qwen3.5-flash",
            api_key=os.getenv("QWEN_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # ✅ 通过 bind 注入 temperature 等调用参数
        self.model = base_model.bind(
            temperature=temperature,
            # 可同时绑定其他参数
            top_p=1.0,
            # max_tokens=2048,
        )

class LLM_QWEN3_6_PLUS:
    def __init__(self, temperature: float = 0.0):
        # 先获取基础模型实例（不传 temperature）
        base_model = _build_chat_model(
            model="qwen3.6-plus",
            api_key=os.getenv("QWEN_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # ✅ 通过 bind 注入 temperature 等调用参数
        self.model = base_model.bind(
            temperature=temperature,
            # 可同时绑定其他参数
            top_p=1.0,
            # max_tokens=2048,
        )

class LLM_QWEN_VL_MAX:
    def __init__(self, temperature: float = 0.0):
        # 先获取基础模型实例（不传 temperature）
        base_model = _build_chat_model(
            model="qwen-vl-max",
            api_key=os.getenv("QWEN_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # ✅ 通过 bind 注入 temperature 等调用参数
        self.model = base_model.bind(
            temperature=temperature,
            # 可同时绑定其他参数
            top_p=1.0,
            # max_tokens=2048,
        )


class QWEN_VL_EMBEDDING:
    """阿里云百炼 qwen3-vl-embedding 多模态嵌入（文本/图像）。"""

    def __init__(self):
        self._api_key = os.getenv("QWEN_API_KEY", "")
        self._base_url = (
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
            "multimodal-embedding/multimodal-embedding"
        )
        self.model_name = "qwen3-vl-embedding"

    def _call(
        self,
        contents: list[dict],
        parameters: dict | None = None,
    ) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        data: dict = {
            "model": self.model_name,
            "input": {"contents": contents},
        }
        if parameters is not None:
            data["parameters"] = parameters
        resp = _HTTP_SYNC.post(self._base_url, headers=headers, json=data)
        resp.raise_for_status()
        result = resp.json()
        embeddings = result["output"]["embeddings"]
        return [e["embedding"] for e in embeddings]

    def embed_text(self, text: str) -> list[float]:
        embs = self._call([{"text": text}])
        return embs[0]

    def embed_image(self, image_base64: str) -> list[float]:
        embs = self._call([{"image": image_base64}])
        return embs[0]

    def embed_multimodal(self, text: str, image_base64: str) -> list[float]:
        """文本 + 图像联合 embedding（单次请求多 content）。"""
        embs = self._call([{"text": text}, {"image": image_base64}])
        if len(embs) == 1:
            return embs[0]
        if len(embs) >= 2:
            import numpy as np

            a = np.asarray(embs[0], dtype=np.float32)
            b = np.asarray(embs[1], dtype=np.float32)
            for vec in (a, b):
                n = float(np.linalg.norm(vec))
                if n > 0:
                    vec /= n
            fused = a + b
            nf = float(np.linalg.norm(fused))
            if nf > 0:
                fused /= nf
            return fused.astype(np.float32).tolist()
        raise RuntimeError(f"multimodal embedding 返回空列表: len={len(embs)}")


# 统一入口：切换模型只需修改此行（如 LLM_moonshot()）
# llm = LLM_QWEN3_5_FLASH()
llm = LLM_deepseek()
# llm = LLM_ecnu_max() # 需在 .env 中配置 ECNU_API_KEY
# llm = LLM_moonshot_kimi() # 需在 .env 中配置 MOONSHOT_API_KEY
# llm = LLM_siliconflow_minimax()
# llm = LLM_siliconflow_qwen()

llm_mem = LLM_deepseek()

# 多模态 / 视觉（需在 .env 配置 QWEN_API_KEY，并在百炼控制台开通 qwen3-vl-flash）
# vlm = LLM_QWEN_VL_PLUS()
# vlm = LLM_QWEN3_5_FLASH()
# vlm = LLM_moonshot_kimi()
# vlm = LLM_QWEN3_6_PLUS()
vlm = LLM_QWEN_VL_MAX()
# vlm = LLM_QWEN3_VL_8B_INSTRUCT()  # SiliconFlow，不传 enable_thinking
# vlm = LLM_QWEN3_5_9B()

# 多模态嵌入（module_2/sementic_topk.py 使用，2560 维）
vlm_embedding = QWEN_VL_EMBEDDING()


def get_vlm_model_name() -> str:
    """返回当前 ``vlm`` 实例使用的 model 名称（用于 runs 结果文件命名等）。"""
    m = vlm.model
    for _ in range(8):
        model_name = getattr(m, "model_name", None)
        if model_name:
            return str(model_name)
        model_str = getattr(m, "model", None)
        if isinstance(model_str, str) and model_str:
            return model_str
        bound = getattr(m, "bound", None)
        if bound is not None:
            m = bound
            continue
        break
    return type(vlm).__name__


def slug_for_run_filename(value: str) -> str:
    """将 agent/model 名称转为安全文件名片段（保留字母数字、点、连字符）。"""
    value = value.replace("/", "-")
    slug = re.sub(r"[^\w.\-]+", "-", value).strip("-")
    return slug or "unknown"