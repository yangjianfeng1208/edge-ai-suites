import json
from pathlib import Path
import base64
import numpy as np
import fitz
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import cv2

BASE_DIR = Path(__file__).parent


class AnnotationServer(SimpleHTTPRequestHandler):
    pages_data = []
    annotations = {}
    output_path = None
    pdf_source = None

    def do_GET(self):
        if self.path == '/' or self.path == '/annotate':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self.wfile.write(self.get_html().encode('utf-8'))

        elif self.path.startswith('/page/'):
            path_part = self.path.split('/')[-1].split('?')[0]
            page_num = int(path_part)
            self.send_page_image(page_num)

        elif self.path == '/annotations':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.end_headers()
            self.wfile.write(json.dumps(self.annotations).encode('utf-8'))

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            self.annotations = data

            template_data = {
                'pdf_source': str(self.pdf_source),
                'total_pages': len(self.pages_data),
                'total_questions': len(self.annotations),
                'questions': self.annotations
            }

            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'success', 'saved': len(self.annotations)}
            self.wfile.write(json.dumps(response).encode('utf-8'))

        else:
            self.send_error(404)

    def send_page_image(self, page_num):
        if 0 < page_num <= len(self.pages_data):
            page_data = self.pages_data[page_num - 1]
            img = page_data['image']

            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_bytes = buffer.tobytes()

            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Content-length', len(img_bytes))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            self.wfile.write(img_bytes)
        else:
            self.send_error(404)

    def get_html(self):
        return f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>试卷模板标注工具</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .controls {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        .controls button {{
            padding: 8px 16px;
            margin-right: 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .btn-primary {{ background: #4CAF50; color: white; }}
        .btn-secondary {{ background: #2196F3; color: white; }}
        .btn-danger {{ background: #f44336; color: white; }}
        .btn-primary:hover {{ background: #45a049; }}
        .btn-secondary:hover {{ background: #0b7dda; }}
        .btn-danger:hover {{ background: #da190b; }}
        .canvas-container {{
            position: relative;
            display: inline-block;
            border: 2px solid #ddd;
            margin-top: 10px;
        }}
        canvas {{
            cursor: crosshair;
            display: block;
        }}
        .annotations-list {{
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        .annotation-item {{
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-left: 3px solid #4CAF50;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .status {{
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            display: none;
        }}
        .status.success {{ background: #d4edda; color: #155724; display: block; }}
        .status.error {{ background: #f8d7da; color: #721c24; display: block; }}
        input {{
            padding: 6px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>试卷模板标注工具</h1>

        <div style="padding: 10px; background: #e3f2fd; border-radius: 5px; margin-bottom: 15px;">
            <strong>当前PDF:</strong> {AnnotationServer.pdf_source}
        </div>

        <div class="controls">
            <label>当前页:
                <select id="pageSelect" onchange="loadPage()">
                    {self.generate_page_options()}
                </select>
            </label>
            <button class="btn-secondary" onclick="prevPage()">← 上一页</button>
            <button class="btn-secondary" onclick="nextPage()">下一页 →</button>
            <button class="btn-primary" onclick="saveAnnotations()">💾 保存标注</button>
            <button class="btn-danger" onclick="clearCurrent()">🗑 清除当前</button>
        </div>

        <div id="status" class="status"></div>

        <div class="canvas-container">
            <canvas id="canvas" width="800" height="1000"></canvas>
        </div>

        <div class="annotations-list">
            <h3>已标注题目 (<span id="count">0</span>)</h3>
            <div id="annotationsList"></div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let currentPage = 1;
        const totalPages = {len(self.pages_data)};
        let annotations = {{}};
        let drawing = false;
        let startX, startY;
        let currentImage = null;
        let scale = 1;

        function loadPage() {{
            currentPage = parseInt(document.getElementById('pageSelect').value);

            const img = new Image();
            img.onload = function() {{
                const maxWidth = 1200;
                const maxHeight = 1400;
                scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);

                canvas.width = img.width * scale;
                canvas.height = img.height * scale;

                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                currentImage = img;
                redrawAnnotations();
            }};
            img.src = '/page/' + currentPage + '?t=' + Date.now();
        }}

        function prevPage() {{
            if (currentPage > 1) {{
                currentPage--;
                document.getElementById('pageSelect').value = currentPage;
                loadPage();
            }}
        }}

        function nextPage() {{
            if (currentPage < totalPages) {{
                currentPage++;
                document.getElementById('pageSelect').value = currentPage;
                loadPage();
            }}
        }}

        canvas.addEventListener('mousedown', (e) => {{
            const rect = canvas.getBoundingClientRect();
            startX = (e.clientX - rect.left) / scale;
            startY = (e.clientY - rect.top) / scale;
            drawing = true;
        }});

        canvas.addEventListener('mousemove', (e) => {{
            if (!drawing) return;

            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / scale;
            const y = (e.clientY - rect.top) / scale;

            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
            redrawAnnotations();

            ctx.strokeStyle = 'blue';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX * scale, startY * scale, (x - startX) * scale, (y - startY) * scale);
        }});

        canvas.addEventListener('mouseup', (e) => {{
            if (!drawing) return;
            drawing = false;

            const rect = canvas.getBoundingClientRect();
            const endX = (e.clientX - rect.left) / scale;
            const endY = (e.clientY - rect.top) / scale;

            const x1 = Math.min(startX, endX);
            const y1 = Math.min(startY, endY);
            const x2 = Math.max(startX, endX);
            const y2 = Math.max(startY, endY);

            if (x2 - x1 > 20 && y2 - y1 > 20) {{
                const questionId = prompt('请输入题号 (如 Q1, Q5):');
                if (questionId) {{
                    annotations[questionId] = {{
                        page: currentPage,
                        bbox: [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)],
                        width: Math.round(x2 - x1),
                        height: Math.round(y2 - y1)
                    }};
                    updateAnnotationsList();
                    redrawAnnotations();
                }}
            }}

            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
            redrawAnnotations();
        }});

        function redrawAnnotations() {{
            for (const [qId, anno] of Object.entries(annotations)) {{
                if (anno.page === currentPage) {{
                    const [x1, y1, x2, y2] = anno.bbox;
                    ctx.strokeStyle = 'green';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
                    ctx.fillStyle = 'green';
                    ctx.font = '16px Arial';
                    ctx.fillText(qId, x1 * scale, y1 * scale - 5);
                }}
            }}
        }}

        function updateAnnotationsList() {{
            const list = document.getElementById('annotationsList');
            const count = document.getElementById('count');

            list.innerHTML = '';
            count.textContent = Object.keys(annotations).length;

            for (const [qId, anno] of Object.entries(annotations)) {{
                const div = document.createElement('div');
                div.className = 'annotation-item';
                div.innerHTML = `
                    <span><strong>${{qId}}</strong> - 第${{anno.page}}页 (${{anno.width}}x${{anno.height}})</span>
                    <button class="btn-danger" onclick="deleteAnnotation('${{qId}}')">删除</button>
                `;
                list.appendChild(div);
            }}
        }}

        function deleteAnnotation(qId) {{
            delete annotations[qId];
            updateAnnotationsList();
            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
            redrawAnnotations();
        }}

        function clearCurrent() {{
            for (const qId in annotations) {{
                if (annotations[qId].page === currentPage) {{
                    delete annotations[qId];
                }}
            }}
            updateAnnotationsList();
            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
            redrawAnnotations();
        }}

        function saveAnnotations() {{
            fetch('/save', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(annotations)
            }})
            .then(r => r.json())
            .then(data => {{
                showStatus('success', `✓ 已保存 ${{data.saved}} 道题的标注`);
            }})
            .catch(err => {{
                showStatus('error', '保存失败: ' + err);
            }});
        }}

        function showStatus(type, message) {{
            const status = document.getElementById('status');
            status.className = 'status ' + type;
            status.textContent = message;
            setTimeout(() => {{ status.style.display = 'none'; }}, 3000);
        }}

        loadPage();
    </script>
</body>
</html>
        '''

    def generate_page_options(self):
        options = []
        for i in range(1, len(self.pages_data) + 1):
            options.append(f'<option value="{i}">第 {i} 页</option>')
        return '\n'.join(options)


def load_pdf(pdf_path):
    print("正在加载PDF...")
    pdf_doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        mat = fitz.Matrix(300/72, 300/72)
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
            'image': img
        })

    pdf_doc.close()
    print(f"已加载 {len(pages)} 页")
    return pages


def main():
    TEST_DATA_DIR = BASE_DIR / "test_data" / "2025_sh_zhongkao_yuwen"
    OUTPUT_DIR = BASE_DIR / "templates"

    pdf_path = TEST_DATA_DIR / "original_test_paper" / "empty_paper.pdf"

    if not pdf_path.exists():
        print(f"错误: 找不到PDF文件 {pdf_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "paper_template.json"

    AnnotationServer.pdf_source = str(pdf_path.resolve())
    AnnotationServer.pages_data = load_pdf(pdf_path)
    AnnotationServer.output_path = output_path

    print("\n" + "="*60)
    print("试卷模板标注工具已启动")
    print("="*60)
    print(f"\n请在浏览器中打开: http://localhost:8000/annotate")
    print("\n操作说明:")
    print("  1. 鼠标拖拽框选答题区域")
    print("  2. 输入题号 (如 Q1, Q5)")
    print("  3. 点击'保存标注'保存到文件")
    print("  4. Ctrl+C 停止服务器")
    print("="*60)
    print()

    server = HTTPServer(('localhost', 8000), AnnotationServer)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
        print(f"标注已保存到: {output_path}")


if __name__ == '__main__':
    main()
