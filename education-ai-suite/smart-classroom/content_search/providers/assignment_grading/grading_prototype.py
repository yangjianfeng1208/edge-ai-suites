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
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import yaml
import numpy as np
import requests
from paddleocr import PaddleOCR
from PIL import Image

from utils.image_preprocessing import ImagePreprocessor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        self.concurrent_workers = self.config.get('concurrent_workers', 2)


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

                # 调试：检查finish_reason
                choice = result['choices'][0]
                content = choice['message']['content']
                finish_reason = choice.get('finish_reason', 'unknown')

                if finish_reason != 'stop':
                    logger.warning(f"VLM finish_reason={finish_reason}, content may be truncated!")
                    logger.warning(f"Full response: {json.dumps(result, ensure_ascii=False, indent=2)}")

                return content
            except Exception as e:
                logger.warning(f"VLM request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise RuntimeError("VLM request failed after all retries")


class QuestionSegmenter:
    QUESTION_PATTERNS = [
        (r'^\d+\.', 'standard'),           # "1." or "1. " (空格可选)
        (r'^\d+、', 'chinese_pause'),       # "1、" (中文顿号)
        (r'^\d+\)', 'parenthesis'),         # "1)" or "1) "
        (r'^（\d+）', 'chinese_paren'),     # "（1）"
        (r'^\(\d+\)', 'english_paren'),     # "(1)" or "(1) "
    ]

    @staticmethod
    def extract_questions(ocr_result: List) -> List[Dict]:
        if not ocr_result or not ocr_result[0]:
            logger.warning("OCR result is empty or None")
            return []

        logger.info(f"OCR returned {len(ocr_result[0])} text lines")

        questions = []
        for idx, line in enumerate(ocr_result[0]):
            if not line or len(line) < 2:
                continue

            text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
            bbox = line[0]

            logger.debug(f"Line {idx}: '{text}'")

            matched = False
            for pattern, pattern_name in QuestionSegmenter.QUESTION_PATTERNS:
                match = re.match(pattern, text)
                if match:
                    numbers = re.findall(r'\d+', match.group())
                    if numbers:
                        q_number = int(numbers[0])

                        if not QuestionSegmenter._is_valid_question_number(text, q_number):
                            logger.debug(f"  → Rejected: '{text[:30]}' (not a valid question)")
                            continue

                        questions.append({
                            'number': q_number,
                            'bbox': bbox,
                            'text': text,
                            'pattern': pattern_name,
                            'ocr_line_index': idx  # 记录OCR行索引
                        })
                        logger.info(f"✓ Matched question {q_number} with pattern '{pattern_name}': {text[:50]}")
                        matched = True
                        break

            if not matched and any(c.isdigit() for c in text[:10]):
                logger.debug(f"  → No match for text starting with: '{text[:30]}'")

        # 计算每题的答题区域（从当前题到下一题之间）
        for i, q in enumerate(questions):
            if i < len(questions) - 1:
                # 不是最后一题：答题区域到下一题开始
                next_q = questions[i + 1]
                q['answer_region_end_y'] = int(min([p[1] for p in next_q['bbox']]))
            else:
                # 最后一题：答题区域到图片底部（或使用合理估计）
                q['answer_region_end_y'] = None

        logger.info(f"Total questions extracted: {len(questions)}")
        return questions

    @staticmethod
    def _is_valid_question_number(text: str, q_number: int) -> bool:
        """
        验证是否是真正的题号

        排除规则：
        1. 题号过大（>30，一般试卷不会超过30题）
        2. 文本太短（<8个字符，可能只是页码或其他标记）
        3. 小数点开头（如"2.7米"这种测量值）
        """
        if q_number > 30:
            logger.debug(f"    Reject reason: q_number {q_number} > 30")
            return False

        if len(text) < 6:
            logger.debug(f"    Reject reason: text too short (len={len(text)})")
            return False

        if re.match(r'^\d+\.\d+', text):
            logger.debug(f"    Reject reason: starts with decimal number")
            return False

        return True


class ChoiceGrader:
    @staticmethod
    def extract_answer_from_question_text(question_text: str) -> str:
        """
        从选择题题干中提取学生答案

        格式：1. 下列代数式中，计算正确的是（A）
        提取括号内的字母：A、B、C、D
        """
        # 匹配括号内的单个大写字母（中文括号或英文括号）
        patterns = [
            r'[（(]\s*([A-D])\s*[)）]',  # （A）或 (A)
        ]

        for pattern in patterns:
            match = re.search(pattern, question_text)
            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def grade(question: Dict, answer_key: Dict, ocr) -> Dict:
        q_num = str(question['number'])
        if q_num not in answer_key:
            return {'score': 0, 'error': 'Question not in answer key'}

        key = answer_key[q_num]
        expected = key['answer']

        try:
            # 首先从题干文本中提取答案（选择题答案在题干末尾的括号里）
            student_answer = ChoiceGrader.extract_answer_from_question_text(question['text'])

            # 如果题干中没找到，尝试从OCR结果中查找
            if not student_answer and hasattr(ocr, 'last_ocr_result') and ocr.last_ocr_result:
                start_idx = question.get('ocr_line_index', 0)
                next_q = question.get('_next_question')
                end_idx = next_q.get('ocr_line_index', len(ocr.last_ocr_result[0])) if next_q else len(ocr.last_ocr_result[0])

                # 只检查当前题的OCR行（不包括下方选项）
                for i in range(start_idx, min(start_idx + 2, end_idx)):  # 最多看2行
                    if i >= len(ocr.last_ocr_result[0]):
                        break
                    line = ocr.last_ocr_result[0][i]
                    if not line or len(line) < 2:
                        continue
                    text = line[1][0] if isinstance(line[1], tuple) else str(line[1])

                    answer = ChoiceGrader.extract_answer_from_question_text(text)
                    if answer:
                        student_answer = answer
                        break

            logger.debug(f"Q{q_num} choice answer extracted: '{student_answer}'")

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
    def extract_answer_from_ocr(question: Dict, ocr_result: list) -> str:
        """
        从OCR结果中提取答题文本

        策略：
        1. 从当前题的OCR行开始，到下一题之前的所有行
        2. 尝试从题干文本中提取（如 =___答案）
        3. 只提取简短的答案部分
        """
        if not ocr_result or not ocr_result[0]:
            return ""

        start_idx = question.get('ocr_line_index', 0)
        next_q = question.get('_next_question')
        end_idx = next_q.get('ocr_line_index', len(ocr_result[0])) if next_q else len(ocr_result[0])

        all_texts = []
        for i in range(start_idx, end_idx):
            if i >= len(ocr_result[0]):
                break
            line = ocr_result[0][i]
            if not line or len(line) < 2:
                continue
            text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
            all_texts.append(text)

        full_text = ' '.join(all_texts)

        # 尝试从题干中提取答案（填空题）
        # 策略：等号/横线后的内容，直到遇到下一题号或行尾
        patterns = [
            # =_答案 或 =___答案（等号+横线后的内容）
            (r'=[_\s]*(.+?)(?:\s+\d+[.、]|$)', 30),
            # "为 答案" 或 "为___答案"
            (r'为[_\s]*(.+?)(?:\s+\d+[.、]|$)', 25),
            # "是 答案"
            (r'是[_\s]*(.+?)(?:\s+\d+[.、]|$)', 25),
        ]

        for pattern, max_len in patterns:
            match = re.search(pattern, full_text)
            if match:
                answer = match.group(1).strip()

                # 只取第一个有效部分（遇到这些符号就截断）
                # 方括号、大括号、"第一题卷"等标记表示后面不是答案了
                stop_markers = [
                    r'\s+[a-z][-+]\d',      # x-1, y+2 (可能是下一题的公式)
                    r'\s*[\[\{【]',         # 方括号/大括号
                    r'\s+(Shijuan|Com|提供|第.{0,5}卷)',  # 网站水印/标记
                ]

                for marker in stop_markers:
                    parts = re.split(marker, answer, maxsplit=1)
                    if len(parts) > 1:
                        answer = parts[0].strip()

                # 限制长度
                if answer and 0 < len(answer) <= max_len:
                    return answer

        return ""

    @staticmethod
    def grade(question: Dict, answer_key: Dict, ocr, vlm_client, image: np.ndarray) -> Dict:
        q_num = str(question['number'])
        if q_num not in answer_key:
            return {'score': 0, 'error': 'Question not in answer key'}

        key = answer_key[q_num]
        expected = key['answer']

        try:
            # 首先尝试从OCR结果中提取答案（包含题干和答题区域）
            if hasattr(ocr, 'last_ocr_result') and ocr.last_ocr_result:
                student_answer = BlankGrader.extract_answer_from_ocr(question, ocr.last_ocr_result)
                logger.debug(f"Q{q_num} extracted from OCR: '{student_answer}'")
            else:
                logger.debug(f"Q{q_num} no last_ocr_result available")
                student_answer = ""

            # 如果题干中没有找到答案，再尝试从答题区域OCR识别
            if not student_answer:
                y_min = int(max([p[1] for p in question['bbox']]))  # 题干底部
                if question.get('answer_region_end_y'):
                    y_max = question['answer_region_end_y']
                else:
                    y_max = min(y_min + 60, image.shape[0])  # 最后一题或+60px

                answer_region = image[y_min:y_max, :]

                # 保存答题区域图片用于调试
                if logger.level <= logging.DEBUG:
                    debug_path = f"outputs/q{q_num}_answer_region.jpg"
                    cv2.imwrite(debug_path, answer_region)

                ocr_result = ocr.ocr(answer_region)
                if ocr_result and ocr_result[0]:
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
            # 提取完整的题目区域（包含题干和答题部分）
            y_min = int(min([p[1] for p in question['bbox']]))  # 题干顶部

            # 向上扩展60像素以包含可能的标题（如"三、解答题"）
            y_min = max(0, y_min - 60)

            if question.get('answer_region_end_y'):
                y_max = question['answer_region_end_y']
            else:
                y_max = image.shape[0]  # 到图片底部

            # 完整题目区域：从题干开始到下一题（或图片底部）
            full_question_region = image[y_min:y_max, :]

            # 保存完整题目区域用于调试
            if logger.level <= logging.DEBUG:
                debug_path = f"outputs/q{q_num}_full_question.jpg"
                cv2.imwrite(debug_path, full_question_region)

            prompt = f"""看图片中的第{q_num}题，满分{max_score}分。

评分并说明理由。"""

            response = vlm_client.chat(prompt, images=[full_question_region], max_tokens=2000)

            # 保存VLM完整响应到文件（开发调试用）
            vlm_debug_path = f"outputs/q{q_num}_vlm_response.txt"
            with open(vlm_debug_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"Question {q_num} VLM Response\n")
                f.write("=" * 60 + "\n\n")
                f.write(response)
                f.write("\n\n" + "=" * 60 + "\n")
            logger.info(f"Q{q_num} VLM response saved to: {vlm_debug_path}")

            # 同时打印到控制台
            logger.info(f"Q{q_num} VLM full response:\n{response}")

            # 移除markdown代码块标记
            response_cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()

            # 从自然语言回复中提取分数和评语
            score = None
            comment = ""

            # 提取分数的多种模式
            score_patterns = [
                r'(?:得|给|评|应得)\s*(\d+)\s*分',  # "给8分"
                r'(?:满分|总分|分数)[：:]\s*(\d+)',  # "满分：8"
                r'(\d+)\s*/\s*' + str(max_score),  # "8/8"
                r'\*\*(\d+)分\*\*',  # markdown粗体
            ]

            for pattern in score_patterns:
                matches = re.findall(pattern, response_cleaned)
                for match in matches:
                    try:
                        s = int(match)
                        if 0 <= s <= max_score:
                            score = s
                            break
                    except:
                        continue
                if score is not None:
                    break

            # 如果没找到明确分数，根据关键词推断
            if score is None:
                if '满分' in response_cleaned or '完全正确' in response_cleaned:
                    score = max_score
                    comment = "VLM评价：满分"
                elif '正确' in response_cleaned and '错误' not in response_cleaned:
                    score = int(max_score * 0.9)
                    comment = "VLM评价：基本正确"
                else:
                    # 查找任何0-max_score的数字
                    numbers = re.findall(r'\b(\d+)\b', response_cleaned)
                    for num in numbers:
                        n = int(num)
                        if 0 <= n <= max_score:
                            score = n
                            break

            # 提取评语（取前100字符）
            if not comment:
                comment = response_cleaned[:100].replace('\n', ' ')

            if score is None:
                score = 0
                logger.warning(f"Q{q_num} 无法提取分数，默认为0")

            score = min(max(score, 0), max_score)
            logger.info(f"Q{q_num} VLM: score={score}/{max_score}, comment={comment[:50]}")

            return {
                'score': score,
                'max_score': max_score,
                'comment': comment,
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
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

        self.ocr = PaddleOCR(
            lang=config.ocr_config['lang'],
            use_textline_orientation=False,
            enable_mkldnn=False,
            use_gpu=False
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

            assignment_id = Path(image_path).stem

            corrected_image, debug_img = ImagePreprocessor.auto_correct_paper(
                image, debug=True
            )

            if corrected_image is not None:
                preprocessed_path = f"outputs/{assignment_id}_preprocessed.jpg"
                cv2.imwrite(preprocessed_path, corrected_image)
                logger.info(f"Saved preprocessed image: {preprocessed_path}")

                if debug_img is not None:
                    debug_path = f"outputs/{assignment_id}_debug.jpg"
                    cv2.imwrite(debug_path, debug_img)
                    logger.info(f"Saved debug image: {debug_path}")

                processing_image = corrected_image
            else:
                logger.warning(f"Paper edge detection failed for {image_path}, using original image")
                processing_image = image

            ocr_result = self.ocr.ocr(processing_image)

            # 将OCR结果存储到OCR对象中，供答题提取使用
            self.ocr.last_ocr_result = ocr_result

            ocr_debug_path = f"outputs/{assignment_id}_ocr_result.json"
            if ocr_result and ocr_result[0]:
                ocr_texts = [
                    {
                        "text": line[1][0] if isinstance(line[1], tuple) else str(line[1]),
                        "confidence": line[1][1] if isinstance(line[1], tuple) else 1.0,
                        "bbox": [[int(p[0]), int(p[1])] for p in line[0]]
                    }
                    for line in ocr_result[0] if line and len(line) >= 2
                ]
                with open(ocr_debug_path, 'w', encoding='utf-8') as f:
                    json.dump(ocr_texts, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved OCR debug info: {ocr_debug_path}")

            questions = QuestionSegmenter.extract_questions(ocr_result)

            logger.info(f"Found {len(questions)} questions in {Path(image_path).name}")

            results = []
            for i, q in enumerate(questions):
                # 传递下一题信息用于确定答题区域边界
                next_q = questions[i + 1] if i < len(questions) - 1 else None
                q['_next_question'] = next_q

            for q in questions:
                q_num = q['number']
                q_type = self.config.question_type_map.get(q_num, 'unknown')

                if q_type == 'choice':
                    result = ChoiceGrader.grade(q, self.config.answer_key, self.ocr)
                elif q_type == 'blank':
                    result = BlankGrader.grade(q, self.config.answer_key, self.ocr, self.vlm_client, processing_image)
                elif q_type == 'calculation':
                    result = CalculationGrader.grade(q, self.config.answer_key, self.vlm_client, processing_image)
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
