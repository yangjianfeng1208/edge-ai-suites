import cv2
import numpy as np
import fitz


def render_pdf_to_images(pdf_path, dpi=300):
    pdf_doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        if pix.n == 4:
            img = img.reshape(pix.h, pix.w, 4)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = img.reshape(pix.h, pix.w, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        pages.append({
            'page_num': page_num + 1,
            'image': img,
            'width': pix.width,
            'height': pix.height
        })

    pdf_doc.close()
    return pages
