import argparse
import io
import base64
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn
from paddleocr_vl_service import PaddleOCRVLService

ocr_service: Optional[PaddleOCRVLService] = None
server_config = {
    'model_path': None,
    'device': None,
    'port': None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_service
    model_path = server_config['model_path']
    device = server_config['device']
    port = server_config['port']

    print(f"\n{'='*80}")
    print(f"Starting PaddleOCR-VL Server...")
    print(f"  Model Path: {model_path}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")

    try:
        ocr_service = PaddleOCRVLService(model_path=model_path, device=device)
        print(f"\n{'='*80}")
        print(f"Server Ready!")
        print(f"  API Docs: http://localhost:{port}/docs")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}\n")
        raise

    yield

    print("\nShutting down OCR service...")


app = FastAPI(
    title="PaddleOCR-VL API Server",
    description="OCR service with support for text, tables, formulas, and charts",
    version="1.0.0",
    lifespan=lifespan
)


class OCRRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    task: str = Field(default="ocr", description="Task type: ocr, table, formula, chart")
    max_new_tokens: int = Field(default=4096, ge=512, le=8192, description="Maximum tokens to generate")
    max_pixels: Optional[int] = Field(default=10000000, ge=1000000, le=20000000, description="Maximum image pixels")


class OCRResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    inference_time: Optional[float] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": ocr_service.device
    }


@app.get("/stats")
async def get_stats():
    """Get performance statistics"""
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    stats = ocr_service.get_perf_stats()
    page_times = stats['page_times']

    return {
        "model_load_time": stats['model_load_time'],
        "total_pages_processed": len(page_times),
        "avg_inference_time": sum(page_times) / len(page_times) if page_times else 0,
        "min_inference_time": min(page_times) if page_times else 0,
        "max_inference_time": max(page_times) if page_times else 0
    }


@app.post("/ocr/file", response_model=OCRResponse)
async def ocr_file(
    file: UploadFile = File(..., description="Image file to process"),
    task: str = Form(default="ocr", description="Task type: ocr, table, formula, chart"),
    max_new_tokens: int = Form(default=4096, ge=512, le=8192),
    max_pixels: Optional[int] = Form(default=10000000, ge=1000000, le=20000000)
):
    """
    OCR endpoint that accepts image file upload

    Supported formats: JPG, JPEG, PNG, BMP, TIFF
    """
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    if task not in ["ocr", "table", "formula", "chart"]:
        raise HTTPException(status_code=400, detail=f"Invalid task type: {task}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        import time
        start_time = time.time()

        text = ocr_service.ocr_image(
            image,
            task=task,
            max_new_tokens=max_new_tokens,
            max_pixels=max_pixels
        )

        inference_time = time.time() - start_time

        return OCRResponse(
            success=True,
            text=text,
            inference_time=inference_time
        )

    except Exception as e:
        return OCRResponse(
            success=False,
            error=str(e)
        )


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(request: OCRRequest):
    """
    OCR endpoint that accepts base64 encoded image

    Request body:
    {
        "image_base64": "base64_encoded_image_string",
        "task": "ocr",
        "max_new_tokens": 4096,
        "max_pixels": 10000000
    }
    """
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    if request.task not in ["ocr", "table", "formula", "chart"]:
        raise HTTPException(status_code=400, detail=f"Invalid task type: {request.task}")

    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))

        import time
        start_time = time.time()

        text = ocr_service.ocr_image(
            image,
            task=request.task,
            max_new_tokens=request.max_new_tokens,
            max_pixels=request.max_pixels
        )

        inference_time = time.time() - start_time

        return OCRResponse(
            success=True,
            text=text,
            inference_time=inference_time
        )

    except Exception as e:
        return OCRResponse(
            success=False,
            error=str(e)
        )


def main():
    config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    default_model = cfg['model']['path']
    default_device = cfg['model']['device']
    default_host = cfg['server']['host']
    default_port = cfg['server']['port']

    parser = argparse.ArgumentParser(description="PaddleOCR-VL API Server")
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Path to OpenVINO model directory (default: {default_model})"
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help=f"OpenVINO device (default: {default_device})"
    )
    parser.add_argument(
        "--host",
        default=default_host,
        help=f"Host to bind the server (default: {default_host})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port to bind the server (default: {default_port})"
    )

    args = parser.parse_args()

    model_path = Path(__file__).parent / args.model
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        return

    server_config['model_path'] = str(model_path)
    server_config['device'] = args.device
    server_config['port'] = args.port

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
