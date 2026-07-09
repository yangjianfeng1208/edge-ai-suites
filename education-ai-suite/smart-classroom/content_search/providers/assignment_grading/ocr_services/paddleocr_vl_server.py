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
    'port': None,
    'llm_int4_compress': False,
    'llm_int8_compress': True,
    'vision_int8_quant': False,
    'llm_int8_quant': True
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_service
    model_path = server_config['model_path']
    device = server_config['device']
    port = server_config['port']
    llm_int4 = server_config['llm_int4_compress']
    llm_int8 = server_config['llm_int8_compress']
    vision_int8 = server_config['vision_int8_quant']
    llm_int8_quant = server_config['llm_int8_quant']

    print(f"\n{'='*80}")
    print(f"Starting PaddleOCR-VL Server...")
    print(f"  Model Path: {model_path}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")

    try:
        ocr_service = PaddleOCRVLService(
            model_path=model_path,
            device=device,
            llm_int4_compress=llm_int4,
            llm_int8_compress=llm_int8,
            vision_int8_quant=vision_int8,
            llm_int8_quant=llm_int8_quant
        )
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


class Region(BaseModel):
    bbox: list[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    type: str = Field(default="text", description="Region type: text, table, formula, chart")
    region_id: str = Field(..., description="Unique region identifier")


class RegionResult(BaseModel):
    region_id: str
    type: str
    bbox: list[float]
    content: str
    inference_time: float


class RegionsOCRResponse(BaseModel):
    success: bool
    results: Optional[list[RegionResult]] = None
    error: Optional[str] = None
    total_inference_time: Optional[float] = None
    num_regions: Optional[int] = None


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


@app.post("/ocr/regions", response_model=RegionsOCRResponse)
async def ocr_regions(
    file: UploadFile = File(..., description="Full page image file"),
    regions: str = Form(..., description="JSON array of regions to process"),
    max_new_tokens: int = Form(default=4096, ge=512, le=8192),
    max_pixels: Optional[int] = Form(default=10000000, ge=1000000, le=20000000)
):
    """
    OCR endpoint that processes multiple regions in a single image

    This is the standard two-stage approach:
    1. Layout detection (PP-DocLayout) identifies regions
    2. Region-based OCR (this endpoint) processes each region with appropriate task

    Args:
        file: Full page image
        regions: JSON string array of regions, e.g.:
            [
                {"bbox": [x1,y1,x2,y2], "type": "text", "region_id": "r1"},
                {"bbox": [x1,y1,x2,y2], "type": "table", "region_id": "r2"}
            ]
        max_new_tokens: Maximum tokens per region
        max_pixels: Maximum pixels per region

    Returns:
        RegionsOCRResponse with results for each region
    """
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    try:
        import time
        import json

        # Parse regions JSON
        try:
            regions_list = json.loads(regions)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid regions JSON: {e}")

        if not isinstance(regions_list, list):
            raise HTTPException(status_code=400, detail="regions must be a JSON array")

        # Load full image
        contents = await file.read()
        full_image = Image.open(io.BytesIO(contents))

        print(f"\n{'='*80}")
        print(f"Processing {len(regions_list)} regions from image")
        print(f"  Image size: {full_image.size}")
        print(f"{'='*80}")

        # Process each region
        results = []
        total_start = time.time()

        # Task mapping
        type_to_task = {
            'text': 'ocr',
            'table': 'table',
            'display_formula': 'formula',
            'inline_formula': 'formula',
            'formula': 'formula',
            'chart': 'chart',
            'paragraph_title': 'ocr',
            'title': 'ocr',
            'doc_title': 'ocr'
        }

        for i, region in enumerate(regions_list, 1):
            try:
                # Validate region
                if not all(k in region for k in ['bbox', 'region_id']):
                    print(f"  [Region {i}] Skipped: missing required fields")
                    continue

                bbox = region['bbox']
                region_type = region.get('type', 'text')
                region_id = region['region_id']

                # Map region type to OCR task
                task = type_to_task.get(region_type, 'ocr')

                # Crop region
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Validate bbox
                if x2 <= x1 or y2 <= y1:
                    print(f"  [Region {i}/{len(regions_list)}] {region_id}: Invalid bbox")
                    continue

                cropped = full_image.crop((x1, y1, x2, y2))

                print(f"  [Region {i}/{len(regions_list)}] {region_id}")
                print(f"    Type: {region_type} → Task: {task}")
                print(f"    BBox: [{x1},{y1},{x2},{y2}]")
                print(f"    Size: {cropped.size}")

                # OCR this region
                region_start = time.time()

                content = ocr_service.ocr_image(
                    cropped,
                    task=task,
                    max_new_tokens=max_new_tokens,
                    max_pixels=max_pixels
                )

                region_time = time.time() - region_start

                print(f"    Time: {region_time:.2f}s")
                print(f"    Content preview: {content[:100]}..." if len(content) > 100 else f"    Content: {content}")

                # Add result
                results.append(RegionResult(
                    region_id=region_id,
                    type=region_type,
                    bbox=bbox,
                    content=content,
                    inference_time=region_time
                ))

            except Exception as e:
                print(f"  [Region {i}] Error: {e}")
                # Continue processing other regions
                continue

        total_time = time.time() - total_start

        print(f"\n{'='*80}")
        print(f"Completed: {len(results)}/{len(regions_list)} regions processed")
        print(f"Total time: {total_time:.2f}s")
        print(f"{'='*80}\n")

        return RegionsOCRResponse(
            success=True,
            results=results,
            total_inference_time=total_time,
            num_regions=len(results)
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return RegionsOCRResponse(
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

    llm_int4 = cfg['model'].get('llm_int4_compress', False)
    llm_int8 = cfg['model'].get('llm_int8_compress', True)
    vision_int8 = cfg['model'].get('vision_int8_quant', False)
    llm_int8_quant = cfg['model'].get('llm_int8_quant', True)

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
    server_config['llm_int4_compress'] = llm_int4
    server_config['llm_int8_compress'] = llm_int8
    server_config['vision_int8_quant'] = vision_int8
    server_config['llm_int8_quant'] = llm_int8_quant

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
