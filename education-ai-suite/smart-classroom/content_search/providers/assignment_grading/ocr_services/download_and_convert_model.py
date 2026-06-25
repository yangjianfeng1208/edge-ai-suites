import os
import shutil
import sys
from pathlib import Path
import argparse

try:
    from ov_paddleocr_vl import PaddleOCR_VL_OV
except ImportError as e:
    print("Error: Failed to import ov_paddleocr_vl")
    print(f"Reason: {e}")
    print("\nMake sure you have installed all dependencies:")
    print("  venv\\Scripts\\activate")
    print("  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def download_pretrained_model(model_id: str, cache_dir: Path) -> Path:
    try:
        from modelscope import snapshot_download
        print(f"Downloading from ModelScope: {model_id}")
        local_dir = Path(snapshot_download(model_id, cache_dir=str(cache_dir)))
        return local_dir
    except Exception as e_modelscope:
        print("ModelScope download not available or failed:")
        print("  ", repr(e_modelscope))

    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
        print(f"Downloading from HuggingFace Hub: {model_id}")
        local_dir = Path(
            hf_snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                local_dir=str(cache_dir / "hf" / model_id.replace("/", "__")),
                local_dir_use_symlinks=False,
            )
        )
        return local_dir
    except Exception as e_hf:
        raise RuntimeError(
            "Failed to download the pretrained model.\n"
            f"- Tried ModelScope repo_id={model_id}\n"
            f"- Tried HuggingFace repo_id={model_id}\n"
            "Install modelscope or huggingface_hub:\n"
            "  pip install modelscope\n"
            "  or\n"
            "  pip install huggingface_hub\n"
        ) from e_hf


def main():
    parser = argparse.ArgumentParser(description="Download and convert PaddleOCR-VL model to OpenVINO")
    parser.add_argument(
        "--model-id",
        default="PaddlePaddle/PaddleOCR-VL-1.6",
        choices=["PaddlePaddle/PaddleOCR-VL-1.6", "PaddlePaddle/PaddleOCR-VL-1.5", "PaddlePaddle/PaddleOCR-VL"],
        help="Model ID to download"
    )
    parser.add_argument("--cache-dir", default="./_cache", help="Cache directory for downloaded models")
    parser.add_argument("--output-dir", help="Output directory for OpenVINO model")
    parser.add_argument("--device", default="CPU", help="OpenVINO device")
    parser.add_argument("--llm-int4", action="store_true", help="Enable INT4 compression for LLM")
    parser.add_argument("--llm-int8", action="store_true", default=True, help="Enable INT8 compression for LLM (default)")
    parser.add_argument("--vision-int8", action="store_true", help="Enable INT8 quantization for vision encoder")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    cache_dir = Path(args.cache_dir).absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model_id
    print(f"Selected model: {model_id}")

    if args.output_dir:
        ov_out_dir = Path(args.output_dir).absolute()
    else:
        ov_out_dir = script_dir / f"ov_{model_id.split('/')[-1].lower().replace('.', '_')}_model"

    patch_modeling_file = script_dir / "modeling_paddleocr_vl.py"
    if not patch_modeling_file.exists():
        print(f"Error: Missing modeling file: {patch_modeling_file}")
        return

    if ov_out_dir.exists():
        print(f"Output directory already exists: {ov_out_dir}")
        overwrite = input("Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Aborted.")
            return
        shutil.rmtree(ov_out_dir)

    print("\n" + "="*80)
    print("Step 1: Downloading pretrained model")
    print("="*80)
    pretrained_dir = download_pretrained_model(model_id, cache_dir)
    print(f"Downloaded to: {pretrained_dir}")

    print("\n" + "="*80)
    print("Step 2: Patching modeling file")
    print("="*80)
    target = pretrained_dir / "modeling_paddleocr_vl.py"
    backup = pretrained_dir / "modeling_paddleocr_vl.py.bak"

    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
        print(f"Backed up original: {backup}")

    shutil.copy2(patch_modeling_file, target)
    print(f"Patched: {target}")

    print("\n" + "="*80)
    print("Step 3: Converting to OpenVINO IR")
    print("="*80)
    print(f"Output directory: {ov_out_dir}")
    print(f"LLM INT4 compress: {args.llm_int4}")
    print(f"LLM INT8 compress: {args.llm_int8}")
    print(f"Vision INT8 quant: {args.vision_int8}")
    print()

    paddleocr_vl_ov = PaddleOCR_VL_OV(
        pretrained_model_path=str(pretrained_dir),
        ov_model_path=str(ov_out_dir),
        device=args.device,
        llm_int4_compress=args.llm_int4,
        llm_int8_compress=args.llm_int8,
        vision_int8_quant=args.vision_int8,
    )

    paddleocr_vl_ov.export_paddleocr_vl_to_ov()
    print("\n" + "="*80)
    print("Conversion complete!")
    print("="*80)
    print(f"Model saved to: {ov_out_dir}")

    paddleocr_vl_ov.close()
    print("Resources released.")


if __name__ == "__main__":
    main()
