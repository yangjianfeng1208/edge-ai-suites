import requests
import base64
from pathlib import Path
import argparse
from PIL import Image
import io


def test_health(base_url):
    """Test health check endpoint"""
    print(f"\n{'='*80}")
    print("Testing /health endpoint...")
    print(f"{'='*80}")

    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_stats(base_url):
    """Test stats endpoint"""
    print(f"\n{'='*80}")
    print("Testing /stats endpoint...")
    print(f"{'='*80}")

    response = requests.get(f"{base_url}/stats")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


def test_ocr_file(base_url, image_path, task="ocr", max_new_tokens=4096, max_pixels=10000000):
    """Test file upload OCR endpoint"""
    print(f"\n{'='*80}")
    print(f"Testing /ocr/file endpoint with {image_path}...")
    print(f"  Task: {task}")
    print(f"  Max Tokens: {max_new_tokens}")
    print(f"  Max Pixels: {max_pixels}")
    print(f"{'='*80}")

    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        data = {
            'task': task,
            'max_new_tokens': max_new_tokens,
            'max_pixels': max_pixels
        }

        response = requests.post(f"{base_url}/ocr/file", files=files, data=data)

    print(f"Status Code: {response.status_code}")
    result = response.json()

    if result['success']:
        print(f"Inference Time: {result['inference_time']:.2f}s")
        print(f"\nOCR Result:")
        print(f"{'-'*80}")
        print(result['text'])
        print(f"{'-'*80}")
    else:
        print(f"ERROR: {result['error']}")

    return result


def test_ocr_base64(base_url, image_path, task="ocr", max_new_tokens=4096, max_pixels=10000000):
    """Test base64 OCR endpoint"""
    print(f"\n{'='*80}")
    print(f"Testing /ocr/base64 endpoint with {image_path}...")
    print(f"  Task: {task}")
    print(f"  Max Tokens: {max_new_tokens}")
    print(f"  Max Pixels: {max_pixels}")
    print(f"{'='*80}")

    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

    payload = {
        'image_base64': image_base64,
        'task': task,
        'max_new_tokens': max_new_tokens,
        'max_pixels': max_pixels
    }

    response = requests.post(f"{base_url}/ocr/base64", json=payload)

    print(f"Status Code: {response.status_code}")
    result = response.json()

    if result['success']:
        print(f"Inference Time: {result['inference_time']:.2f}s")
        print(f"\nOCR Result:")
        print(f"{'-'*80}")
        print(result['text'])
        print(f"{'-'*80}")
    else:
        print(f"ERROR: {result['error']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test PaddleOCR-VL Server")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Server base URL"
    )
    parser.add_argument(
        "--image",
        help="Path to test image"
    )
    parser.add_argument(
        "--task",
        default="ocr",
        choices=["ocr", "table", "formula", "chart"],
        help="OCR task type"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=10000000,
        help="Maximum image pixels"
    )
    parser.add_argument(
        "--method",
        choices=["file", "base64", "both"],
        default="file",
        help="Test method: file upload or base64"
    )

    args = parser.parse_args()

    base_url = args.url.rstrip('/')

    if not test_health(base_url):
        print("\nERROR: Server health check failed!")
        return

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"\nERROR: Image not found: {image_path}")
            return

        if args.method in ["file", "both"]:
            test_ocr_file(
                base_url,
                image_path,
                task=args.task,
                max_new_tokens=args.max_tokens,
                max_pixels=args.max_pixels
            )

        if args.method in ["base64", "both"]:
            test_ocr_base64(
                base_url,
                image_path,
                task=args.task,
                max_new_tokens=args.max_tokens,
                max_pixels=args.max_pixels
            )

    test_stats(base_url)

    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
