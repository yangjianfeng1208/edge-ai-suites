#!/usr/bin/env python3
"""
Assignment Grading Prototype
Automated grading system for Math exam assignments using OCR and VLM.
"""

import os
import re
import json
import glob
import base64
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import yaml
import numpy as np
import requests
from paddleocr import PaddleOCR
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_path="config.yaml", answer_key_path="answer_key.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        with open(answer_key_path, 'r', encoding='utf-8') as f:
            self.answer_key = json.load(f)
            if 'comment' in self.answer_key:
                del self.answer_key['comment']

        self.subject = self.config.get('subject', 'Math')
        self.question_type_map = self.config['question_type_map']
        self.vlm_config = self.config['vlm_service']
        self.ocr_config = self.config['ocr_config']
        self.concurrent_workers = self.config.get('concurrent_workers', 4)


class VLMClient:
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 2):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def check_health(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"VLM health check failed: {e}")
            return False

    def chat(self, prompt: str, images: List[np.ndarray] = None, max_tokens: int = 512) -> str:
        encoded_images = []
        if images:
            for img in images:
                _, buffer = cv2.imencode('.jpg', img)
                b64 = base64.b64encode(buffer).decode('utf-8')
                encoded_images.append(f"data:image/jpeg;base64,{b64}")

        payload = {
            "model": "Qwen2.5-VL-3B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [
                        {"type": "image_url", "image_url": {"url": img_url}}
                        for img_url in encoded_images
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                logger.warning(f"VLM request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise RuntimeError("VLM request failed after all retries")


class ImagePreprocessor:
    @staticmethod
    def simple_preprocess(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return gray

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle

        if abs(angle) > 0.5:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                gray, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
        else:
            rotated = gray

        binary = cv2.adaptiveThreshold(
            rotated, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        return binary


class QuestionSegmenter:
    QUESTION_PATTERNS = [
        (r'^\d+\.\s', 'standard'),
        (r'^\d+\)\s', 'parenthesis'),
        (r'^（\d+）', 'chinese_paren'),
        (r'^\(\d+\)\s', 'english_paren'),
    ]

    @staticmethod
    def extract_questions(ocr_result: List) -> List[Dict]:
        if not ocr_result or not ocr_result[0]:
            return []

        questions = []
        for line in ocr_result[0]:
            if not line or len(line) < 2:
                continue

            text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
            bbox = line[0]

            for pattern, pattern_name in QuestionSegmenter.QUESTION_PATTERNS:
                match = re.match(pattern, text)
                if match:
                    numbers = re.findall(r'\d+', match.group())
                    if numbers:
                        q_number = int(numbers[0])
                        questions.append({
                            'number': q_number,
                            'bbox': bbox,
                            'text': text,
                            'pattern': pattern_name
                        })
                        break

        return questions


class ChoiceGrader:
    @staticmethod
    def grade(question: Dict, answer_key: Dict, ocr, image: np.ndarray) -> Dict:
        q_num = str(question['number'])
        if q_num not in answer_key:
            return {'score': 0, 'error': 'Question not in answer key'}

        key = answer_key[q_num]
        expected = key['answer']

        try:
            y_min = int(min([p[1] for p in question['bbox']]))
            y_max = int(max([p[1] for p in question['bbox']]))
            answer_region = image[y_min:y_max + 50, :]

            ocr_result = ocr.ocr(answer_region)
            if not ocr_result or not ocr_result[0]:
                return {'score': 0, 'error': 'No OCR result', 'student_answer': None}

            full_text = ' '.join([line[1][0] for line in ocr_result[0]])
            letters = re.findall(r'\b[A-D]\b', full_text)
            student_answer = letters[0] if letters else None

            correct = (student_answer == expected)

            return {
                'score': key['score'] if correct else 0,
                'max_score': key['score'],
                'correct': correct,
                'student_answer': student_answer,
                'expected': expected,
                'type': 'choice'
            }
        except Exception as e:
            logger.error(f"Error grading choice question {q_num}: {e}")
            return {'score': 0, 'error': str(e), 'student_answer': None}


class BlankGrader:
    @staticmethod
    def grade(question: Dict, answer_key: Dict, ocr, vlm_client, image: np.ndarray) -> Dict:
        q_num = str(question['number'])
        if q_num not in answer_key:
            return {'score': 0, 'error': 'Question not in answer key'}

        key = answer_key[q_num]
        expected = key['answer']

        try:
            y_min = int(min([p[1] for p in question['bbox']]))
            y_max = int(max([p[1] for p in question['bbox']]))
            answer_region = image[y_min:y_max + 50, :]

            ocr_result = ocr.ocr(answer_region)
            if not ocr_result or not ocr_result[0]:
                student_answer = ""
            else:
                student_answer = ' '.join([line[1][0] for line in ocr_result[0]]).strip()

            if 'tolerance' in key and key['tolerance'] is not None:
                try:
                    student_val = float(re.sub(r'[^\d.-]', '', student_answer))
                    expected_val = float(expected)
                    tolerance = key.get('tolerance', 0)

                    correct = abs(student_val - expected_val) <= tolerance
                    return {
                        'score': key['score'] if correct else 0,
                        'max_score': key['score'],
                        'correct': correct,
                        'student_answer': student_answer,
                        'expected': expected,
                        'type': 'blank'
                    }
                except (ValueError, TypeError):
                    pass

            if student_answer.replace(' ', '') == expected.replace(' ', ''):
                return {
                    'score': key['score'],
                    'max_score': key['score'],
                    'correct': True,
                    'student_answer': student_answer,
                    'expected': expected,
                    'type': 'blank'
                }

            return {
                'score': 0,
                'max_score': key['score'],
                'correct': False,
                'student_answer': student_answer,
                'expected': expected,
                'type': 'blank'
            }

        except Exception as e:
            logger.error(f"Error grading blank question {q_num}: {e}")
            return {'score': 0, 'error': str(e)}


class CalculationGrader:
    @staticmethod
    def grade(question: Dict, answer_key: Dict, vlm_client: VLMClient, image: np.ndarray) -> Dict:
        q_num = str(question['number'])
        if q_num not in answer_key:
            return {'score': 0, 'error': 'Question not in answer key'}

        key = answer_key[q_num]
        max_score = key['max_score']

        try:
            y_min = int(min([p[1] for p in question['bbox']]))
            answer_region = image[y_min:, :]

            prompt = f"""You are grading a Math exam calculation/proof question.

Question Number: {q_num}
Maximum Score: {max_score}

Evaluate the student's answer based on:
1. Correct final answer
2. Valid mathematical steps
3. Clear reasoning

Output ONLY a JSON object (no markdown, no extra text):
{{"score": <number 0-{max_score}>, "comment": "<brief comment>"}}"""

            response = vlm_client.chat(prompt, images=[answer_region], max_tokens=200)

            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

            score = min(max(int(result.get('score', 0)), 0), max_score)

            return {
                'score': score,
                'max_score': max_score,
                'comment': result.get('comment', ''),
                'type': 'calculation'
            }

        except Exception as e:
            logger.error(f"Error grading calculation question {q_num}: {e}")
            return {
                'score': 0,
                'max_score': max_score,
                'error': str(e),
                'type': 'calculation'
            }


class AssignmentGrader:
    def __init__(self, config: Config):
        self.config = config

        logger.info("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            lang=config.ocr_config['lang']
        )

        logger.info("Initializing VLM client...")
        self.vlm_client = VLMClient(
            base_url=config.vlm_config['base_url'],
            timeout=config.vlm_config['timeout'],
            max_retries=config.vlm_config['max_retries']
        )

        if not self.vlm_client.check_health():
            raise RuntimeError(f"VLM service not available at {config.vlm_config['base_url']}")

        logger.info("Initialization complete")

    def process_single_assignment(self, image_path: str) -> Dict:
        logger.info(f"Processing: {image_path}")
        start_time = datetime.now()

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            ocr_result = self.ocr.ocr(image)
            questions = QuestionSegmenter.extract_questions(ocr_result)

            logger.info(f"Found {len(questions)} questions in {Path(image_path).name}")

            results = []
            for q in questions:
                q_num = q['number']
                q_type = self.config.question_type_map.get(q_num, 'unknown')

                if q_type == 'choice':
                    result = ChoiceGrader.grade(q, self.config.answer_key, self.ocr, image)
                elif q_type == 'blank':
                    result = BlankGrader.grade(q, self.config.answer_key, self.ocr, self.vlm_client, image)
                elif q_type == 'calculation':
                    result = CalculationGrader.grade(q, self.config.answer_key, self.vlm_client, image)
                else:
                    result = {'score': 0, 'error': f'Unknown question type: {q_type}'}

                result['question_number'] = q_num
                results.append(result)

            total_score = sum(r.get('score', 0) for r in results)
            max_possible = sum(r.get('max_score', 0) for r in results if 'max_score' in r)

            processing_time = (datetime.now() - start_time).total_seconds()

            output = {
                'assignment_id': Path(image_path).stem,
                'total_score': total_score,
                'max_score': max_possible,
                'questions': results,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat()
            }

            output_path = f"outputs/{Path(image_path).stem}_result.json"
            os.makedirs('outputs', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            logger.info(f"Completed: {Path(image_path).name} - Score: {total_score}/{max_possible} - Time: {processing_time:.1f}s")
            return output

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}", exc_info=True)
            return {
                'assignment_id': Path(image_path).stem,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def process_batch(self, image_paths: List[str]) -> List[Dict]:
        logger.info(f"Processing {len(image_paths)} assignments with {self.config.concurrent_workers} workers")

        with ThreadPoolExecutor(max_workers=self.config.concurrent_workers) as executor:
            results = list(executor.map(self.process_single_assignment, image_paths))

        return results


def main():
    logger.info("=" * 60)
    logger.info("Assignment Grading System - Prototype")
    logger.info("=" * 60)

    config = Config()
    logger.info(f"Subject: {config.subject}")
    logger.info(f"Concurrent workers: {config.concurrent_workers}")

    grader = AssignmentGrader(config)

    image_patterns = ['test_data/**/*.jpg', 'test_data/**/*.png', 'test_data/**/*.JPG', 'test_data/**/*.PNG']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern, recursive=True))

    if not image_files:
        logger.error("No image files found in test_data/")
        return

    logger.info(f"Found {len(image_files)} image(s) to process")

    results = grader.process_batch(image_files)

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")

    if successful:
        avg_score = sum(r['total_score'] for r in successful) / len(successful)
        avg_time = sum(r['processing_time_seconds'] for r in successful) / len(successful)
        logger.info(f"Average score: {avg_score:.1f}")
        logger.info(f"Average processing time: {avg_time:.1f}s")

    logger.info("Results saved to outputs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
