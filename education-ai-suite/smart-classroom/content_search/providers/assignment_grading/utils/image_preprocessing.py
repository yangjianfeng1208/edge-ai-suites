"""
OpenCV-based image preprocessing utilities for exam paper OCR
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class ImagePreprocessor:
    """
    Image preprocessing for exam paper OCR optimization
    """

    def __init__(self):
        pass

    @staticmethod
    def detect_paper_edges(image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        检测试卷的四个边角点（标准流程）

        流程：
        1. 灰度化 + 去噪
        2. Canny边缘检测
        3. findContours轮廓检测（RETR_EXTERNAL）
        4. 按面积排序，筛选4个顶点的轮廓
        5. 返回四个角点

        Args:
            image: 输入图像 (BGR格式)
            debug: 是否返回调试图像

        Returns:
            corners: 试卷四个角点坐标 [[左上], [右上], [右下], [左下]]
                    如果检测失败返回 None
        """
        # 1. 灰度化 + 去噪
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # 可选：形态学操作增强边缘连续性
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 3. 查找轮廓（只保留外轮廓）
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            if debug:
                return None, None
            return None

        # 4. 按面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        h, w = image.shape[:2]
        image_area = h * w
        paper_contour = None

        # 5. 筛选：寻找面积大、有4个顶点、接近矩形的轮廓
        for contour in contours[:15]:  # 检查前15个最大轮廓
            area = cv2.contourArea(contour)

            # 面积要求：至少占图片的10%（放宽以支持部分遮挡的情况）
            if area < image_area * 0.10:
                continue

            # 多边形近似
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # 必须是4个顶点
            if len(approx) == 4:
                # 检查是否接近矩形（长宽比合理）
                rect = cv2.minAreaRect(approx)
                box_width, box_height = rect[1]
                if box_width > 0 and box_height > 0:
                    aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
                    # A4纸长宽比约1.4，允许0.7-2.0范围
                    if 0.7 <= aspect_ratio <= 2.0:
                        paper_contour = approx
                        break

        if paper_contour is None:
            if debug:
                return None, None
            return None

        corners = paper_contour.reshape(4, 2)
        corners = ImagePreprocessor._order_points(corners)

        if debug:
            debug_img = image.copy()
            cv2.drawContours(debug_img, [paper_contour], -1, (0, 255, 0), 3)
            for i, corner in enumerate(corners):
                cv2.circle(debug_img, tuple(corner.astype(int)), 10, (0, 0, 255), -1)
                cv2.putText(debug_img, str(i), tuple(corner.astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return corners, debug_img

        return corners

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        按照 [左上, 右上, 右下, 左下] 的顺序排列四个点
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    @staticmethod
    def perspective_transform(image: np.ndarray, corners: np.ndarray,
                             target_width: Optional[int] = None,
                             target_height: Optional[int] = None) -> np.ndarray:
        """
        对试卷进行透视变换，矫正为正视图

        Args:
            image: 输入图像
            corners: 试卷四个角点 [左上, 右上, 右下, 左下]
            target_width: 目标宽度（像素），如果为None则自动计算
            target_height: 目标高度（像素），如果为None则自动计算

        Returns:
            warped: 矫正后的图像
        """
        (tl, tr, br, bl) = corners

        width_bottom = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
        width_top = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
        max_width = max(int(width_bottom), int(width_top))

        height_left = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
        height_right = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
        max_height = max(int(height_left), int(height_right))

        if target_width is None:
            target_width = max_width
        if target_height is None:
            target_height = max_height

        dst_points = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        warped = cv2.warpPerspective(image, matrix, (target_width, target_height))

        return warped

    @staticmethod
    def auto_correct_paper(image: np.ndarray,
                          target_width: Optional[int] = None,
                          target_height: Optional[int] = None,
                          debug: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        自动检测并矫正试卷（边缘检测 + 透视变换）

        Args:
            image: 输入图像
            target_width: 目标宽度
            target_height: 目标高度
            debug: 是否返回调试信息

        Returns:
            (corrected_image, debug_info) 或 (None, None) 如果失败
        """
        if debug:
            corners, debug_img = ImagePreprocessor.detect_paper_edges(image, debug=True)
        else:
            corners = ImagePreprocessor.detect_paper_edges(image, debug=False)
            debug_img = None

        if corners is None:
            return None, debug_img

        warped = ImagePreprocessor.perspective_transform(
            image, corners, target_width, target_height
        )

        return warped, debug_img
