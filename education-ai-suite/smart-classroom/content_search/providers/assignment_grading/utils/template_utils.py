import json
import fitz


def load_template(template_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_pdf_dimensions(pdf_path):
    doc = fitz.open(str(pdf_path))
    dimensions = {}
    for i in range(len(doc)):
        page = doc[i]
        dimensions[i + 1] = {
            'width': page.rect.width,
            'height': page.rect.height
        }
    doc.close()
    return dimensions


def calculate_scale_factors(template, student_pdf_path):
    template_pdf = template['pdf_source']
    template_dims = get_pdf_dimensions(template_pdf)
    student_dims = get_pdf_dimensions(student_pdf_path)

    scale_factors = {}
    for page_num in template_dims:
        if page_num in student_dims:
            scale_x = student_dims[page_num]['width'] / template_dims[page_num]['width']
            scale_y = student_dims[page_num]['height'] / template_dims[page_num]['height']
            scale_factors[page_num] = {'x': scale_x, 'y': scale_y}

    return scale_factors


def scale_bbox(bbox, scale_x, scale_y):
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]


def find_nearest_anchors(question_bbox, template_anchors, student_anchors, page_num, k=3):
    q_center_x = (question_bbox[0] + question_bbox[2]) / 2
    q_center_y = (question_bbox[1] + question_bbox[3]) / 2

    distances = []
    for q_id, t_anchor in template_anchors.items():
        if t_anchor['page'] != page_num:
            continue
        if q_id not in student_anchors:
            continue

        t_pos = t_anchor['position']
        dist = ((t_pos[0] - q_center_x) ** 2 + (t_pos[1] - q_center_y) ** 2) ** 0.5
        distances.append((dist, q_id))

    distances.sort()
    return [q_id for _, q_id in distances[:k]]


def calculate_global_offset(template_anchors, student_anchors, template_dims, student_dims):
    page_1_scale_x = student_dims[1]['width'] / template_dims[1]['width']
    page_1_scale_y = student_dims[1]['height'] / template_dims[1]['height']

    if 'title_header' in template_anchors and 'title_header' in student_anchors:
        t_pos = template_anchors['title_header']['position']
        s_pos = student_anchors['title_header']['position']

        offset_x = s_pos[0] - t_pos[0] * page_1_scale_x
        offset_y = s_pos[1] - t_pos[1] * page_1_scale_y

        return {
            'offset_x': offset_x,
            'offset_y': offset_y,
            'scale_x': page_1_scale_x,
            'scale_y': page_1_scale_y
        }

    common_ids = set(template_anchors.keys()) & set(student_anchors.keys())
    if len(common_ids) < 3:
        return {'offset_x': 0, 'offset_y': 0, 'scale_x': page_1_scale_x, 'scale_y': page_1_scale_y}

    template_points = []
    student_points = []
    for q_id in sorted(common_ids):
        if not q_id.startswith('corner'):
            template_points.append(template_anchors[q_id]['position'])
            student_points.append(student_anchors[q_id]['position'])

    if len(template_points) < 2:
        return {'offset_x': 0, 'offset_y': 0, 'scale_x': page_1_scale_x, 'scale_y': page_1_scale_y}

    avg_offset_x = sum(s[0] - t[0] * page_1_scale_x for s, t in zip(student_points, template_points)) / len(template_points)
    avg_offset_y = sum(s[1] - t[1] * page_1_scale_y for s, t in zip(student_points, template_points)) / len(template_points)

    return {
        'offset_x': avg_offset_x,
        'offset_y': avg_offset_y,
        'scale_x': page_1_scale_x,
        'scale_y': page_1_scale_y
    }


def calculate_local_transform(question_bbox, nearest_anchors, template_anchors, student_anchors, global_transform):
    if len(nearest_anchors) == 0:
        return global_transform

    if len(nearest_anchors) == 1:
        q_id = nearest_anchors[0]
        t_pos = template_anchors[q_id]['position']
        s_pos = student_anchors[q_id]['position']

        offset_x = s_pos[0] - t_pos[0] * global_transform['scale_x']
        offset_y = s_pos[1] - t_pos[1] * global_transform['scale_y']

        return {
            'type': 'affine',
            'scale_x': global_transform['scale_x'],
            'scale_y': global_transform['scale_y'],
            'offset_x': offset_x,
            'offset_y': offset_y
        }

    template_points = []
    student_points = []
    for q_id in nearest_anchors:
        template_points.append(template_anchors[q_id]['position'])
        student_points.append(student_anchors[q_id]['position'])

    avg_scale_x = sum(s[0] / t[0] for s, t in zip(student_points, template_points)) / len(nearest_anchors)
    avg_scale_y = sum(s[1] / t[1] for s, t in zip(student_points, template_points)) / len(nearest_anchors)

    avg_offset_x = sum(s[0] - t[0] * avg_scale_x for s, t in zip(student_points, template_points)) / len(nearest_anchors)
    avg_offset_y = sum(s[1] - t[1] * avg_scale_y for s, t in zip(student_points, template_points)) / len(nearest_anchors)

    return {
        'type': 'affine',
        'scale_x': avg_scale_x,
        'scale_y': avg_scale_y,
        'offset_x': avg_offset_x,
        'offset_y': avg_offset_y
    }


def apply_local_transform(bbox, transform):
    if not transform:
        return bbox

    x1, y1, x2, y2 = bbox

    if transform['type'] == 'translate':
        return [
            int(x1 + transform['offset_x']),
            int(y1 + transform['offset_y']),
            int(x2 + transform['offset_x']),
            int(y2 + transform['offset_y'])
        ]
    elif transform['type'] == 'affine':
        return [
            int(x1 * transform['scale_x'] + transform['offset_x']),
            int(y1 * transform['scale_y'] + transform['offset_y']),
            int(x2 * transform['scale_x'] + transform['offset_x']),
            int(y2 * transform['scale_y'] + transform['offset_y'])
        ]

    return bbox


def add_corner_anchors(anchors, page_dims):
    corners = {
        'corner_top_left': {'page': 1, 'position': (100, 100)},
        'corner_top_right': {'page': 1, 'position': (page_dims[1]['width'] - 100, 100)},
        'corner_bottom_left': {'page': 1, 'position': (100, page_dims[1]['height'] - 100)},
        'corner_bottom_right': {'page': 1, 'position': (page_dims[1]['width'] - 100, page_dims[1]['height'] - 100)},
    }

    for page_num in range(2, len(page_dims) + 1):
        corners[f'corner_top_left_p{page_num}'] = {'page': page_num, 'position': (100, 100)}
        corners[f'corner_top_right_p{page_num}'] = {'page': page_num, 'position': (page_dims[page_num]['width'] - 100, 100)}

    anchors.update(corners)
    return anchors


def extract_answer_region(page_image, bbox):
    x1, y1, x2, y2 = bbox
    h, w = page_image.shape[:2]

    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return page_image[y1:y2, x1:x2]
