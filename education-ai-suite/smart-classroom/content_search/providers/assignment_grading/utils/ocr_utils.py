import re
from typing import List, Dict, Tuple
from paddleocr import PaddleOCR

_ocr_instance = None


def get_ocr_engine():
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(lang='ch')
    return _ocr_instance


def extract_question_numbers(page_image, skip_until_content_start=True) -> List[Dict]:
    ocr = get_ocr_engine()
    result = ocr.ocr(page_image, cls=False)

    questions = []
    content_started = not skip_until_content_start

    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            bbox = line[0]

            if not content_started:
                if any(keyword in text for keyword in ['一、', '文言文阅读', '默写']):
                    content_started = True
                continue

            patterns = [
                r'^(\d+)[.、]',
                r'^\((\d+)\)',
            ]

            for pattern in patterns:
                match = re.match(pattern, text.strip())
                if match:
                    q_num = int(match.group(1))
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    questions.append({
                        'number': q_num,
                        'text': text.strip(),
                        'bbox': (int(min(x_coords)), int(min(y_coords)),
                                int(max(x_coords)), int(max(y_coords))),
                        'confidence': confidence
                    })
                    break

    return questions


def calculate_answer_region(question: Dict, next_question: Dict, page_height: int,
                            margin_top: int = 10, margin_bottom: int = 10) -> Tuple[int, int, int, int]:
    y_start = question['bbox'][3] + margin_top

    if next_question:
        y_end = next_question['bbox'][1] - margin_bottom
    else:
        y_end = page_height

    x_start = 0
    x_end = question.get('page_width', 10000)

    return (x_start, y_start, x_end, y_end)


def recognize_text_from_image(image) -> str:
    ocr = get_ocr_engine()
    result = ocr.ocr(image, cls=False)

    if not result or not result[0]:
        return ""

    texts = []
    for line in result[0]:
        text = line[1][0]
        texts.append(text)

    return ' '.join(texts)


def detect_title_anchor(image, page_num: int) -> Dict:
    ocr = get_ocr_engine()
    result = ocr.ocr(image, cls=False)

    if not result or not result[0]:
        return {}

    for line in result[0]:
        text = line[1][0].strip()
        bbox = line[0]

        if '2025' in text and ('上海' in text or '中考' in text or '语文' in text):
            y_coords = [p[1] for p in bbox]
            x_coords = [p[0] for p in bbox]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)

            return {
                'title_header': {
                    'page': page_num,
                    'position': (center_x, center_y),
                    'bbox': bbox
                }
            }

    return {}


def detect_question_number_anchors(image, page_num: int) -> Dict:
    ocr = get_ocr_engine()
    result = ocr.ocr(image, cls=False)

    anchors = {}
    if not result or not result[0]:
        return anchors

    import re
    patterns = [
        r'^(\d+)[.、]',
        r'^\((\d+)\)',
    ]

    for line in result[0]:
        text = line[1][0].strip()
        bbox = line[0]

        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                q_num = int(match.group(1))
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]

                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)

                anchors[f'Q{q_num}'] = {
                    'page': page_num,
                    'position': (center_x, center_y),
                    'bbox': bbox
                }
                break

    return anchors
