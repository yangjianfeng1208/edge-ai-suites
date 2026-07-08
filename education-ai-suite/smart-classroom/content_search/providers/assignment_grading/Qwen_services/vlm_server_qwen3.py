"""VLM API Server using Qwen3-VL with OpenVINO GenAI

Qwen3-VL features:
- Enhanced OCR: 32 languages, robust in low light/blur/tilt
- Advanced spatial perception and reasoning
- Long context (256K native, expandable to 1M)
- Superior multimodal reasoning (STEM/Math)

Model download/conversion:
1. Install: pip install optimum-intel nncf
2. Convert: optimum-cli export openvino --model Qwen/Qwen3-VL-4B-Instruct --task image-text-to-text qwen3_vl_4b_int4 --weight-format int4
3. Move to: models/openvino/Qwen3-VL-4B-int4/
"""

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union
from contextlib import asynccontextmanager
import base64
import io
import sys
import os
import logging
import time
import traceback
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)7s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Qwen3-VL 模型路径 (支持环境变量覆盖)
MODEL_NAME = os.getenv("QWEN3_MODEL", "Qwen3-VL-8B-Instruct-int4-ov")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", MODEL_NAME)
MODEL_PATH = os.path.normpath(MODEL_PATH)

DEVICE = os.getenv("VLM_DEVICE", "GPU.1")
PORT = int(os.getenv("VLM_PORT", "9900"))

vlm_pipeline = None
current_device = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="VLM API Server (Qwen3-VL)",
    description="OpenAI-compatible API for Qwen3-VL-4B with enhanced OCR and reasoning",
    version="1.0.0",
    lifespan=lifespan
)


class ImageURL(BaseModel):
    url: str


class ContentText(BaseModel):
    type: str = "text"
    text: str


class ContentImage(BaseModel):
    type: str = "image_url"
    image_url: ImageURL


class Message(BaseModel):
    role: str
    content: Union[str, List[Union[ContentText, ContentImage]]]


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen3-VL-4B-int4"
    messages: List[Message]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    model_path: str


def load_model():
    """加载 Qwen3-VL 模型"""
    global vlm_pipeline, current_device

    logger.info("=" * 60)
    logger.info(f"Model Loading - {MODEL_NAME}")
    logger.info("=" * 60)
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Target device: {DEVICE}")

    # 检查模型目录
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model directory not found: {MODEL_PATH}")
        logger.error("")
        logger.error("Please download and convert Qwen3-VL model:")
        logger.error("  1. pip install optimum-intel nncf")
        logger.error("  2. optimum-cli export openvino \\")
        logger.error("       --model Qwen/Qwen3-VL-4B-Instruct \\")
        logger.error("       --task image-text-to-text \\")
        logger.error("       qwen3_vl_4b_int4 \\")
        logger.error("       --weight-format int4")
        logger.error(f"  3. Move to: {MODEL_PATH}")
        logger.error("")
        sys.exit(1)

    # 检查关键文件
    required_files = [
        "openvino_language_model.xml",
        "openvino_language_model.bin",
        "openvino_vision_embeddings_model.xml",
        "openvino_detokenizer.xml"
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(MODEL_PATH, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        logger.error("This may not be a complete Qwen3-VL OpenVINO model")
        sys.exit(1)

    ov_cache_dir = os.path.join(SCRIPT_DIR, "ov_cache_qwen3")
    os.makedirs(ov_cache_dir, exist_ok=True)

    ov_config = {
        "CACHE_DIR": ov_cache_dir,
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": "f16",
    }

    try:
        logger.info(f"Loading model on {DEVICE}...")
        logger.info(f"Cache directory: {ov_cache_dir}")
        start_time = time.time()

        vlm_pipeline = ov_genai.VLMPipeline(MODEL_PATH, DEVICE, **ov_config)
        current_device = DEVICE

        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully on {DEVICE} ({load_time:.1f}s)")

    except Exception as e:
        if DEVICE.startswith("GPU"):
            logger.warning(f"Failed to load on GPU: {e}")
            logger.info("Falling back to CPU...")

            try:
                start_time = time.time()
                vlm_pipeline = ov_genai.VLMPipeline(MODEL_PATH, "CPU")
                current_device = "CPU"
                load_time = time.time() - start_time
                logger.info(f"Model loaded successfully on CPU ({load_time:.1f}s)")
            except Exception as cpu_error:
                logger.error(f"Failed to load on CPU: {cpu_error}")
                traceback.print_exc()
                sys.exit(1)
        else:
            logger.error(f"Failed to load model: {e}")
            traceback.print_exc()
            sys.exit(1)

    logger.info("=" * 60)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if vlm_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        device=current_device,
        model_path=MODEL_PATH
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    start_time = time.time()

    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        user_message = None
        for msg in reversed(request.messages):
            if msg.role == 'user':
                user_message = msg
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        content = user_message.content
        if isinstance(content, str):
            content = [ContentText(type="text", text=content)]

        prompt_text = ""
        images = []

        for item in content:
            if isinstance(item, ContentText) or (isinstance(item, dict) and item.get('type') == 'text'):
                prompt_text = item.text if isinstance(item, ContentText) else item.get('text', '')
            elif isinstance(item, ContentImage) or (isinstance(item, dict) and item.get('type') == 'image_url'):
                image_url = item.image_url.url if isinstance(item, ContentImage) else item.get('image_url', {}).get('url', '')
                if image_url.startswith('data:image'):
                    try:
                        image_data = image_url.split(',', 1)[1]
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                        MAX_SIZE = 1920
                        if max(image.size) > MAX_SIZE:
                            ratio = MAX_SIZE / max(image.size)
                            new_size = (int(image.width * ratio), int(image.height * ratio))
                            image = image.resize(new_size, Image.Resampling.LANCZOS)
                            logger.info(f"Resized image: {image.size}")

                        images.append(image)
                        logger.debug(f"Decoded image: {image.size}")
                    except Exception as img_error:
                        logger.error(f"Failed to decode image: {img_error}")
                        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(img_error)}")

        if not prompt_text:
            raise HTTPException(status_code=400, detail="No text prompt found")

        logger.info(f"Request: prompt={len(prompt_text)} chars, images={len(images)}, max_tokens={request.max_tokens}")

        gen_start = time.time()

        from openvino_genai import GenerationConfig

        gen_config = GenerationConfig()
        gen_config.max_new_tokens = request.max_tokens
        gen_config.do_sample = False

        if images:
            # Qwen3-VL expects image as ov.Tensor with shape (1, H, W, 3)
            image_array = np.array(images[0])
            logger.debug(f"Image array shape: {image_array.shape}")

            image_tensor = ov.Tensor(image_array[None])

            result = vlm_pipeline.generate(
                prompt_text,
                image=image_tensor,
                generation_config=gen_config
            )
        else:
            result = vlm_pipeline.generate(
                prompt_text,
                generation_config=gen_config
            )

        response_text = str(result)

        finish_reason = "stop"
        try:
            if hasattr(result, 'finish_reason'):
                finish_reason = str(result.finish_reason)
                logger.info(f"VLM finish_reason: {finish_reason}")
        except Exception as e:
            logger.debug(f"Could not get finish_reason: {e}")

        gen_time = time.time() - gen_start
        total_time = time.time() - start_time

        logger.info(f"Response: {len(response_text)} chars, gen_time={gen_time:.2f}s, total_time={total_time:.2f}s")

        prompt_tokens = max(len(prompt_text) // 2, 1)
        completion_tokens = max(len(response_text) // 2, 1)

        return ChatCompletionResponse(
            id="chatcmpl-" + os.urandom(12).hex(),
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason=finish_reason
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    print("=" * 60)
    print(f"VLM API Server - {MODEL_NAME}")
    print("=" * 60)
    print(f"Starting server on http://127.0.0.1:{PORT}")
    print(f"")
    print(f"Model:  {MODEL_NAME}")
    print(f"Path:   {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"")
    print(f"Features:")
    print(f"  - Enhanced OCR (32 languages)")
    print(f"  - Advanced spatial reasoning")
    print(f"  - Long context (256K)")
    print(f"")
    print(f"Endpoints:")
    print(f"  - GET  /health")
    print(f"  - POST /v1/chat/completions")
    print(f"  - GET  /docs")
    print(f"")
    print(f"Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=PORT,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
