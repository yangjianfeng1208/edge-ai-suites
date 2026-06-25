import http.server
import socketserver
import json
import urllib.parse
from pathlib import Path
import base64
import io
from PIL import Image

BASE_DIR = Path(__file__).parent
PORT = 8001


class AdjustDetectionsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self.send_html_interface()
        elif self.path.startswith('/api/load_detection'):
            self.load_detection_json()
        elif self.path.startswith('/api/page_image/'):
            self.serve_page_image()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/save_detection':
            self.save_detection_json()
        else:
            self.send_error(404)

    def send_html_interface(self):
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>调整YOLO检测结果</title>
    <style>
        body { margin: 0; font-family: Arial; background: #f5f5f5; }
        #toolbar {
            background: #2c3e50;
            color: white;
            padding: 15px;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        #toolbar button {
            padding: 8px 15px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        #toolbar button:hover { background: #2980b9; }
        #toolbar select {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #bdc3c7;
        }
        #container {
            display: flex;
            height: calc(100vh - 60px);
        }
        #canvas-container {
            flex: 1;
            overflow: auto;
            background: #ecf0f1;
            position: relative;
        }
        #canvas {
            cursor: crosshair;
            display: block;
            margin: 20px auto;
        }
        #sidebar {
            width: 300px;
            background: white;
            padding: 20px;
            overflow-y: auto;
            border-left: 1px solid #bdc3c7;
        }
        .detection-item {
            padding: 10px;
            margin: 10px 0;
            background: #ecf0f1;
            border-radius: 4px;
            cursor: pointer;
        }
        .detection-item:hover { background: #d5dbdb; }
        .detection-item.selected {
            background: #3498db;
            color: white;
        }
        .info {
            background: #e8f8f5;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 12px;
        }
        #status {
            color: #27ae60;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="toolbar">
        <span style="font-size: 18px; font-weight: bold;">调整YOLO检测结果</span>
        <select id="pageSelect"></select>
        <select id="classSelect">
            <option value="Answer_Block">Answer_Block</option>
            <option value="Question_Block">Question_Block</option>
            <option value="Question_Answer_Block">Question_Answer_Block</option>
            <option value="Question_Paper_Area">Question_Paper_Area</option>
            <option value="Instruction">Instruction</option>
            <option value="Description">Description</option>
        </select>
        <button onclick="zoomOut()">缩小 (-)</button>
        <button onclick="zoomIn()">放大 (+)</button>
        <span id="zoomLevel" style="margin: 0 10px;">缩放: 50%</span>
        <button onclick="deleteSelected()">删除选中</button>
        <button onclick="saveDetections()">保存JSON</button>
        <span id="status"></span>
    </div>

    <div id="container">
        <div id="canvas-container">
            <canvas id="canvas"></canvas>
        </div>
        <div id="sidebar">
            <div class="info">
                <strong>操作说明：</strong><br>
                1. 选择类别后，拖拽鼠标添加检测框<br>
                2. 点击检测框选中<br>
                3. 按Delete键或点击"删除"按钮删除<br>
                4. 完成后点击"保存JSON"
            </div>
            <h3>检测列表</h3>
            <div id="detectionList"></div>
        </div>
    </div>

    <script>
        let detectionData = null;
        let currentPage = 1;
        let detections = {};
        let selectedDetection = null;
        let isDrawing = false;
        let startX, startY;
        let currentBox = null;
        let scale = 0.5;
        let originalImage = null;

        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const pageSelect = document.getElementById('pageSelect');
        const classSelect = document.getElementById('classSelect');

        async function loadData() {
            const resp = await fetch('/api/load_detection');
            detectionData = await resp.json();
            detections = detectionData.detections;

            Object.keys(detections).forEach(page => {
                const option = document.createElement('option');
                option.value = page;
                option.text = `第${page}页`;
                pageSelect.appendChild(option);
            });

            pageSelect.onchange = () => {
                currentPage = parseInt(pageSelect.value);
                loadPage();
            };

            loadPage();
        }

        async function loadPage() {
            const img = new Image();
            img.src = `/api/page_image/${currentPage}`;
            img.onload = () => {
                originalImage = img;
                redrawCanvas();
            };
        }

        function redrawCanvas() {
            if (!originalImage) return;

            canvas.width = originalImage.width * scale;
            canvas.height = originalImage.height * scale;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
            drawDetections();
            updateDetectionList();
        }

        function zoomIn() {
            scale = Math.min(scale + 0.1, 2.0);
            document.getElementById('zoomLevel').textContent = `缩放: ${Math.round(scale * 100)}%`;
            redrawCanvas();
        }

        function zoomOut() {
            scale = Math.max(scale - 0.1, 0.2);
            document.getElementById('zoomLevel').textContent = `缩放: ${Math.round(scale * 100)}%`;
            redrawCanvas();
        }

        function drawDetections() {
            const pageDetections = detections[currentPage] || [];

            pageDetections.forEach((det, idx) => {
                const [x1, y1, x2, y2] = det.bbox;
                const isSelected = selectedDetection && selectedDetection.page === currentPage && selectedDetection.index === idx;

                const sx1 = x1 * scale;
                const sy1 = y1 * scale;
                const sx2 = x2 * scale;
                const sy2 = y2 * scale;

                ctx.strokeStyle = isSelected ? '#e74c3c' : '#3498db';
                ctx.lineWidth = isSelected ? 3 : 2;
                ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

                ctx.fillStyle = isSelected ? '#e74c3c' : '#3498db';
                ctx.font = `${Math.round(14 * scale)}px Arial`;
                ctx.fillText(det.class_name, sx1, sy1 - 5);
            });
        }

        function updateDetectionList() {
            const list = document.getElementById('detectionList');
            const pageDetections = detections[currentPage] || [];

            list.innerHTML = '';
            pageDetections.forEach((det, idx) => {
                const div = document.createElement('div');
                div.className = 'detection-item';
                if (selectedDetection && selectedDetection.page === currentPage && selectedDetection.index === idx) {
                    div.className += ' selected';
                }
                div.textContent = `${det.class_name} (${det.confidence.toFixed(2)})`;
                div.onclick = () => {
                    selectedDetection = {page: currentPage, index: idx};
                    loadPage();
                };
                list.appendChild(div);
            });
        }

        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
            isDrawing = true;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            redrawCanvas();
            ctx.strokeStyle = '#2ecc71';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX, startY, x - startX, y - startY);
        });

        canvas.addEventListener('mouseup', (e) => {
            if (!isDrawing) return;
            isDrawing = false;

            const rect = canvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;

            const x1 = Math.min(startX, endX) / scale;
            const y1 = Math.min(startY, endY) / scale;
            const x2 = Math.max(startX, endX) / scale;
            const y2 = Math.max(startY, endY) / scale;

            if (Math.abs(x2 - x1) > 10 && Math.abs(y2 - y1) > 10) {
                const className = classSelect.value;
                const classId = getClassId(className);

                if (!detections[currentPage]) {
                    detections[currentPage] = [];
                }

                detections[currentPage].push({
                    class_id: classId,
                    class_name: className,
                    confidence: 1.0,
                    bbox: [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)]
                });

                redrawCanvas();
            }
        });

        function getClassId(className) {
            const mapping = {
                'Answer_Block': 0,
                'Description': 1,
                'Instruction': 2,
                'Question_Answer_Block': 3,
                'Question_Block': 4,
                'Question_Paper_Area': 5
            };
            return mapping[className] || 0;
        }

        function deleteSelected() {
            if (!selectedDetection) return;
            detections[selectedDetection.page].splice(selectedDetection.index, 1);
            selectedDetection = null;
            redrawCanvas();
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Delete') {
                deleteSelected();
            }
        });

        async function saveDetections() {
            detectionData.detections = detections;

            const resp = await fetch('/api/save_detection', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(detectionData)
            });

            const result = await resp.json();
            document.getElementById('status').textContent = result.message;
            setTimeout(() => {
                document.getElementById('status').textContent = '';
            }, 3000);
        }

        loadData();
    </script>
</body>
</html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def load_detection_json(self):
        json_path = BASE_DIR / "outputs/yolo_detections/shuxue_yolo_detections.json"

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def serve_page_image(self):
        page_num = int(self.path.split('/')[-1])

        from utils.pdf_utils import render_pdf_to_images

        # pdf_path = BASE_DIR / "test_data/2025_sh_zhongkao_yuwen/papers/xiaoming/yuwen_paper.pdf"
        pdf_path = BASE_DIR / "test_data/2025_sh_zhongkao_math/2025_sh_zhongkao_math.pdf"
        pages = render_pdf_to_images(pdf_path, dpi=300)

        page_img = [p for p in pages if p['page_num'] == page_num][0]['image']

        import cv2
        _, buffer = cv2.imencode('.jpg', page_img)

        self.send_response(200)
        self.send_header('Content-type', 'image/jpeg')
        self.end_headers()
        self.wfile.write(buffer.tobytes())

    def save_detection_json(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        json_path = BASE_DIR / "outputs/yolo_detections/shuxue_yolo_detections_adjusted.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        result = {'status': 'success', 'message': f'已保存到 {json_path.name}'}

        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))


def main():
    print(f"\n{'='*80}")
    print("YOLO检测结果调整工具")
    print(f"{'='*80}")

    port = PORT
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            httpd = socketserver.TCPServer(("", port), AdjustDetectionsHandler)
            break
        except OSError:
            if attempt < max_attempts - 1:
                port += 1
            else:
                print(f"Error: Unable to find available port after {max_attempts} attempts")
                return

    print(f"\n1. 浏览器打开: http://localhost:{port}")
    print(f"2. 在每页上添加/删除/调整检测框")
    print(f"3. 点击'保存JSON'保存修改")
    print(f"\n输出文件: outputs/yolo_detections/shuxue_yolo_detections_adjusted.json")
    print(f"\n按 Ctrl+C 停止服务器\n")

    with httpd:
        httpd.serve_forever()


if __name__ == '__main__':
    main()
