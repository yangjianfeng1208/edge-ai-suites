import re
import json
from pathlib import Path


def normalize_question_id(question_id, question_mapping=None):
    """
    统一题号格式：Q1, Q1_1, Q2_1 等 -> 数字

    Args:
        question_id: 题号
        question_mapping: 可选的映射字典 {"Q1_1": "1", "Q2_1": "7", ...}

    Examples:
        "Q1_1" (with mapping) -> "1"
        "Q1" -> "1"
        "Q1_1" (no mapping) -> "1"
        "Q2_1" (no mapping) -> "1"
        "1" -> "1"
    """
    if isinstance(question_id, int):
        return str(question_id)

    question_id = str(question_id).strip()

    # 优先使用question_mapping
    if question_mapping and question_id in question_mapping:
        return question_mapping[question_id]

    # 如果已经是纯数字，直接返回
    if question_id.isdigit():
        return question_id

    # 处理 Q1_1 格式：提取下划线后的数字
    if '_' in question_id:
        parts = question_id.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            return parts[1]

    # 处理 Q1 格式：去掉Q前缀
    if question_id.startswith('Q') and question_id[1:].isdigit():
        return question_id[1:]

    return question_id


def parse_objective_answers_from_ocr(ocr_text_path, answer_key_path):
    """
    从OCR文本中提取客观题答案（选择题+填空题）

    Args:
        ocr_text_path: OCR文本文件路径
        answer_key_path: 标准答案JSON路径

    Returns:
        dict: {question_number: student_answer}
    """
    with open(ocr_text_path, 'r', encoding='utf-8') as f:
        ocr_text = f.read()

    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key_data = json.load(f)

    # 提取answers部分
    answer_key = answer_key_data.get('answers', answer_key_data)

    student_answers = {}

    for q_num, q_info in answer_key.items():
        if q_num in ['comment', 'metadata', '_fields']:
            continue

        q_type = q_info.get('type')

        if q_type == 'choice':
            student_answers[q_num] = extract_choice_answer(ocr_text, q_num)
        elif q_type == 'blank':
            student_answers[q_num] = extract_blank_answer(ocr_text, q_num)

    return student_answers


def extract_choice_answer(ocr_text, q_num):
    """
    提取选择题答案
    格式: "1. 下列代数式中，计算正确的是（A）"
    """
    pattern = rf'^{q_num}\.\s+.*?[（(]([A-D])[)）]'
    match = re.search(pattern, ocr_text, re.MULTILINE)

    if match:
        return match.group(1)
    return None


def extract_blank_answer(ocr_text, q_num):
    """
    提取填空题答案
    实际格式: "9. ... 是 \\(\\underline{\\text{m < -\\frac{1}{8}}}\\)"
    实际格式: "11. ... 的解为 \\(\\alpha = 10\\)."
    """
    # 查找完整的题目行
    lines = ocr_text.split('\n')
    q_line = None
    for idx, line in enumerate(lines):
        if re.match(rf'^{q_num}\.\s+', line):
            q_line = line
            break

    if not q_line:
        return None

    # 先尝试提取纯文本答案（"为 X 米"）
    text_answer_match = re.search(r'[为是]\s+([\d.]+)\s*[米元人个张只度]', q_line)
    if text_answer_match:
        return text_answer_match.group(1).strip()

    # 提取所有 \(...\) 块
    latex_blocks = re.findall(r'\\\((.+?)\\\)', q_line)

    if not latex_blocks:
        return None

    # 过滤掉备用数据块（包含sin/cos/tan的）
    answer_candidates = []
    for block in latex_blocks:
        # 跳过备用数据（三角函数值）
        if re.search(r'\\(sin|cos|tan|angle)', block):
            continue
        answer_candidates.append(block)

    if not answer_candidates:
        return None

    # 取最后一个候选块
    answer_block = answer_candidates[-1]

    # 清理LaTeX格式
    answer = answer_block

    # 提取 \underline{...} 内容
    underline_match = re.search(r'\\underline\{(.+)\}', answer)
    if underline_match:
        answer = underline_match.group(1)

    # 处理 \frac{A}{B} -> A/B
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)

    # 移除 \text{...}
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)

    # 保留希腊字母：\alpha -> α, \gamma -> γ等
    greek_map = {
        'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
        'epsilon': 'ε', 'theta': 'θ', 'lambda': 'λ', 'mu': 'μ',
        'pi': 'π', 'sigma': 'σ', 'omega': 'ω'
    }
    for latex_name, unicode_char in greek_map.items():
        answer = answer.replace(f'\\{latex_name}', unicode_char)

    # 移除其他LaTeX命令
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)

    # 清理花括号
    answer = re.sub(r'[{}]', '', answer)

    # 清理多余空格
    answer = re.sub(r'\s+', ' ', answer).strip()

    return answer if answer else None


def grade_objective_questions(student_answers, answer_key, verbose=False):
    """
    评分客观题

    Args:
        student_answers: {question_number: student_answer}
        answer_key: 标准答案JSON或其中的answers部分
        verbose: 是否打印详细匹配信息

    Returns:
        dict: {
            'results': {question_number: {...}},
            'total_score': int,
            'max_score': int
        }
    """
    results = {}
    total_score = 0
    max_score = 0

    # 提取answers部分
    if 'answers' in answer_key and 'metadata' in answer_key:
        answer_key_data = answer_key['answers']
    else:
        answer_key_data = answer_key

    # 构建反向映射: alias -> question_number
    alias_to_qnum = {}
    for q_num, q_info in answer_key_data.items():
        if q_num in ['comment', 'metadata', '_fields'] or not isinstance(q_info, dict):
            continue
        alias = q_info.get('alias')
        if alias:
            alias_to_qnum[alias] = q_num

    # 标准化学生答案的题号
    normalized_student_answers = {}
    for q_id, ans in student_answers.items():
        if q_id in alias_to_qnum:
            normalized_id = alias_to_qnum[q_id]
        else:
            normalized_id = normalize_question_id(q_id, {})
        normalized_student_answers[normalized_id] = ans

    def display_width(s):
        width = 0
        for char in str(s):
            if '一' <= char <= '鿿' or '　' <= char <= '〿':
                width += 2
            else:
                width += 1
        return width

    def pad_string(s, target_width):
        s = str(s)
        current_width = display_width(s)
        if current_width >= target_width:
            return s
        return s + ' ' * (target_width - current_width)

    if verbose:
        print(f"\n{'='*100}")
        print("Objective Question Grading Details")
        print(f"{'='*100}")
        header = f"{pad_string('Q#', 8)}{pad_string('Student', 25)}{pad_string('Answer Key', 35)}{pad_string('Mode', 12)}{pad_string('Result', 8)}"
        print(header)
        print(f"{'-'*100}")

    for q_num, q_info in answer_key_data.items():
        if q_num in ['comment', 'metadata', '_fields'] or not isinstance(q_info, dict):
            continue

        q_score = q_info.get('score', 0)
        max_score += q_score

        student_ans = normalized_student_answers.get(q_num)
        correct_ans = q_info.get('answer')
        q_type = q_info.get('type')
        match_mode = q_info.get('match_mode', 'any')
        q_format = q_info.get('format', 'string')

        is_correct = False

        if student_ans is None:
            is_correct = False
        elif q_type == 'choice':
            if match_mode == 'any':
                if isinstance(correct_ans, list):
                    is_correct = student_ans.upper() in [ans.upper() for ans in correct_ans]
                else:
                    is_correct = (student_ans.upper() == str(correct_ans).upper())
            else:
                is_correct = False
        elif q_type == 'blank':
            is_correct = check_blank_answer(student_ans, correct_ans, 0, match_mode, q_format)

        earned_score = q_score if is_correct else 0
        total_score += earned_score

        if verbose:
            student_str = str(student_ans) if student_ans else 'None'
            correct_str = str(correct_ans) if not isinstance(correct_ans, list) else f"{correct_ans[0]}..." if len(correct_ans) > 1 else str(correct_ans[0])
            result_str = "[V]" if is_correct else "[X]"

            row = f"{pad_string(q_num, 8)}{pad_string(student_str, 25)}{pad_string(correct_str, 35)}{pad_string(match_mode, 12)}{pad_string(result_str, 8)}"
            print(row)

        results[q_num] = {
            'student_answer': student_ans,
            'correct_answer': correct_ans,
            'match_mode': match_mode,
            'format': q_format,
            'is_correct': is_correct,
            'score': earned_score,
            'max_score': q_score
        }

    if verbose:
        print(f"{'-'*100}")
        print(f"Total Score: {total_score}/{max_score}")
        print(f"{'='*100}\n")

    return {
        'results': results,
        'total_score': total_score,
        'max_score': max_score
    }


def check_blank_answer(student_ans, correct_ans, tolerance, match_mode='any', answer_format='string'):
    """
    检查填空题答案是否正确

    Args:
        student_ans: 学生答案
        correct_ans: 标准答案（可能是单个或列表）
        tolerance: 数值容差
        match_mode: 匹配模式
        answer_format: 答案格式 'string', 'number', 'latex'
    """
    if student_ans is None:
        return False

    # 根据format选择标准化方式
    if answer_format == 'latex':
        # LaTeX格式：转换为LaTeX再标准化
        student_ans_normalized = normalize_answer(convert_to_latex_format(student_ans))
    else:
        # 其他格式：直接标准化
        student_ans_normalized = normalize_answer(student_ans)

    # 处理不同的匹配模式
    if match_mode == 'any':
        # 任意一个匹配即可
        if not isinstance(correct_ans, list):
            correct_ans = [correct_ans]

        for ans in correct_ans:
            if answer_format == 'latex':
                ans_normalized = normalize_answer(ans)  # 标准答案已经是LaTeX格式
            else:
                ans_normalized = normalize_answer(ans)

            if student_ans_normalized == ans_normalized:
                return True

            # 数值比较
            if answer_format == 'number':
                try:
                    if abs(float(student_ans_normalized) - float(ans_normalized)) <= tolerance:
                        return True
                except (ValueError, TypeError):
                    pass
        return False

    elif match_mode == 'all':
        # 必须匹配所有答案（顺序）
        # 暂未实现，需要学生答案也是列表
        return False

    elif match_mode == 'set':
        # 集合匹配（无序）
        # 暂未实现，需要学生答案也是列表
        return False

    else:  # exact (默认)
        # 单答案精确匹配
        if isinstance(correct_ans, list):
            # 兼容旧格式
            correct_ans_list = [normalize_answer(ans) for ans in correct_ans]
            return student_ans_normalized in correct_ans_list

        correct_ans_normalized = normalize_answer(correct_ans)

        if student_ans_normalized == correct_ans_normalized:
            return True

        # 数值比较
        try:
            student_num = float(student_ans_normalized)
            correct_num = float(correct_ans_normalized)
            return abs(student_num - correct_num) <= tolerance
        except (ValueError, TypeError):
            pass

        return False


def normalize_answer(answer):
    """
    标准化答案：去除空格、统一符号
    """
    if answer is None:
        return ''

    answer = str(answer).strip()
    answer = answer.replace(' ', '')
    answer = answer.replace('（', '(').replace('）', ')')
    answer = answer.replace('，', ',')
    answer = answer.replace('、', ',')

    return answer


def convert_to_latex_format(answer):
    if answer is None:
        return ''

    answer = str(answer)

    if '\\frac' in answer or '\\sqrt' in answer:
        return answer

    import re
    answer = re.sub(r'([a-zA-Z0-9]+)/([a-zA-Z0-9]+)', r'\\frac{\1}{\2}', answer)

    answer = re.sub(r'\^(\d+)', r'^{\1}', answer)
    answer = re.sub(r'\^(\w)', r'^{\1}', answer)

    greek_reverse = {
        'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta',
        'ε': '\\epsilon', 'θ': '\\theta', 'λ': '\\lambda', 'μ': '\\mu',
        'π': '\\pi', 'σ': '\\sigma', 'ω': '\\omega'
    }
    for unicode_char, latex_name in greek_reverse.items():
        answer = answer.replace(unicode_char, latex_name)

    return answer


def get_objective_score(question_id, ocr_text_path, answer_key_path):
    """
    获取单个客观题的得分

    Args:
        question_id: 题号 (如 "Q1_1" 或 "1")
        ocr_text_path: OCR文本路径
        answer_key_path: 标准答案路径

    Returns:
        dict: {'score': int, 'max_score': int, 'is_correct': bool, 'student_answer': str}
    """
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)

    # 构建alias映射
    alias_to_qnum = {}
    for q_num, q_info in answer_key.items():
        if q_num == 'comment' or not isinstance(q_info, dict):
            continue
        alias = q_info.get('alias')
        if alias:
            alias_to_qnum[alias] = q_num

    # 查找题号
    if question_id in alias_to_qnum:
        normalized_id = alias_to_qnum[question_id]
    else:
        normalized_id = normalize_question_id(question_id, {})

    if normalized_id not in answer_key:
        return {
            'score': 0,
            'max_score': 0,
            'is_correct': False,
            'student_answer': None,
            'error': f'Question {question_id} not in answer_key'
        }

    q_info = answer_key[normalized_id]

    student_answers = parse_objective_answers_from_ocr(ocr_text_path, answer_key_path)
    student_ans = student_answers.get(normalized_id)

    correct_ans = q_info.get('answer')
    q_type = q_info.get('type')
    q_score = q_info.get('score', 0)

    is_correct = False
    if student_ans is not None:
        if q_type == 'choice':
            is_correct = (student_ans.upper() == correct_ans.upper())
        elif q_type == 'blank':
            is_correct = check_blank_answer(student_ans, correct_ans, q_info.get('tolerance', 0))

    return {
        'score': q_score if is_correct else 0,
        'max_score': q_score,
        'is_correct': is_correct,
        'student_answer': student_ans,
        'correct_answer': correct_ans
    }


if __name__ == '__main__':
    ocr_path = Path(__file__).parent.parent / 'outputs/2025_sh_zhongkao_math/ocr_text/2025_sh_zhongkao_math_ocr.txt'
    answer_key_path = Path(__file__).parent.parent / 'test_data/2025_sh_zhongkao_math/rubric_guided_scoring/answer_key.json'

    print("="*80)
    print("客观题自动评分测试")
    print("="*80)

    print(f"\n[1] 从OCR文本提取学生答案: {ocr_path}")
    student_answers = parse_objective_answers_from_ocr(ocr_path, answer_key_path)

    print(f"\n学生答案:")
    for q_num, ans in sorted(student_answers.items(), key=lambda x: int(x[0])):
        print(f"  Q{q_num}: {ans}")

    print(f"\n[2] 评分")
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)

    grading_result = grade_objective_questions(student_answers, answer_key)

    print(f"\n评分结果:")
    for q_num, result in sorted(grading_result['results'].items(), key=lambda x: int(x[0])):
        status = "[V]" if result['is_correct'] else "[X]"
        print(f"  Q{q_num}: {status} 学生答案={result['student_answer']}, 标准答案={result['correct_answer']}, 得分={result['score']}/{result['max_score']}")

    print(f"\n总分: {grading_result['total_score']}/{grading_result['max_score']}")
