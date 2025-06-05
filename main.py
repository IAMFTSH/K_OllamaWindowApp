import sys
import time
import json
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QLineEdit, QComboBox, QProgressBar,
    QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# ====== Model Manager 内联实现 ======
class ModelManager:
    def list_models(self):
        try:
            res = requests.get("http://localhost:11434/api/tags")
            if res.status_code == 200:
                return [m['name'] for m in res.json()['models']]
        except Exception as e:
            print("Error fetching models:", e)
        return []

    def delete_model(self, model_name):
        try:
            res = requests.delete("http://localhost:11434/api/delete", json={"name": model_name})
            return res.status_code == 200
        except Exception as e:
            print("Delete error:", e)
            return False


# ====== Inference Worker 内联实现 ======
class InferenceWorker(QThread):
    update_output = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    finished = pyqtSignal()
    speed_result = pyqtSignal(int, float)

    def __init__(self, model, prompt, temperature, max_tokens, speed_test=False):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.speed_test = speed_test
        self.buffer = ""

    def run(self):
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.model,
            "prompt": self.prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }

        try:
            with requests.post(url, json=data, stream=True) as r:
                if r.status_code != 200:
                    self.update_output.emit("[错误] 请求失败：" + str(r.status_code))
                    return

                total_tokens = 0
                start_time = None

                for line in r.iter_lines():
                    if not line:
                        continue
                    decoded_line = line.decode('utf-8')
                    try:
                        response = json.loads(decoded_line)
                        if 'response' in response:
                            token = response['response']
                            if start_time is None:
                                start_time = time.time()
                            total_tokens += 1

                            self.buffer += token
                            if any(self.buffer.endswith(p) for p in ['。', '！', '？', '.', '!', '?']):
                                self.update_output.emit(self.buffer.strip())
                                self.buffer = ""
                        if 'done' in response and response['done']:
                            if self.buffer:
                                self.update_output.emit(self.buffer.strip())
                                self.buffer = ""
                            end_time = time.time()
                            if self.speed_test:
                                elapsed = end_time - start_time
                                self.speed_result.emit(total_tokens, elapsed)
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.update_output.emit("[错误] " + str(e))

        self.finished.emit()


class DownloadWorker(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        url = "http://localhost:11434/api/pull"
        data = {"name": self.model_name, "stream": True}

        try:
            with requests.post(url, json=data, stream=True) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        response = json.loads(line.decode('utf-8'))
                        total = response.get('total', None)
                        completed = response.get('completed', 0)

                        if total and total > 0:
                            percent = int((completed / total) * 100)
                            self.update_progress.emit(percent)
                        else:
                            progress = response.get('progress', '')
                            if '%' in progress:
                                percent = int(progress.split('%')[0])
                                self.update_progress.emit(percent)
                    except Exception as e:
                        print("Error parsing download progress:", e)
                        continue
            self.finished.emit(True, self.model_name)
        except Exception as e:
            print("Download error:", e)
            self.finished.emit(False, self.model_name)


# ====== Main Window 内联实现 ======
class OllamaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama 模型管理器")
        self.setGeometry(100, 100, 900, 600)

        self.model_manager = ModelManager()

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()

        # 模型选择
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.refresh_button = QPushButton("刷新模型")
        self.refresh_button.clicked.connect(self.load_models)
        model_layout.addWidget(QLabel("模型:"))
        model_layout.addWidget(self.model_combo, 2)
        model_layout.addWidget(self.refresh_button)

        # 参数输入
        param_layout = QHBoxLayout()
        self.temp_input = QLineEdit("0.7")
        self.max_tokens_input = QLineEdit("200")
        param_layout.addWidget(QLabel("温度:"))
        param_layout.addWidget(self.temp_input, 1)
        param_layout.addWidget(QLabel("最大Token数:"))
        param_layout.addWidget(self.max_tokens_input, 1)

        # 输入框
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("请输入你的提示...")

        # 控制按钮
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("运行推理")
        self.run_button.clicked.connect(self.start_inference)
        self.delete_button = QPushButton("删除模型")
        self.delete_button.clicked.connect(self.delete_model)
        self.test_speed_button = QPushButton("测试 Token/s")
        self.test_speed_button.clicked.connect(self.test_token_speed)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.test_speed_button)

        # 模型下载区域
        download_layout = QHBoxLayout()
        self.download_input = QLineEdit()
        self.download_button = QPushButton("下载模型")
        self.download_button.clicked.connect(self.download_model)
        download_layout.addWidget(QLabel("下载模型:"))
        download_layout.addWidget(self.download_input)
        download_layout.addWidget(self.download_button)

        # 输出区域
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # 状态标签（可选）
        self.status_label = QLabel("等待操作...")

        # 布局组合
        layout.addLayout(model_layout)
        layout.addLayout(param_layout)
        layout.addWidget(QLabel("提示输入:"))
        layout.addWidget(self.prompt_input)
        layout.addLayout(button_layout)
        layout.addLayout(download_layout)
        layout.addWidget(QLabel("输出结果:"))
        layout.addWidget(self.output_display)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        self.load_models()

    def load_models(self):
        models = self.model_manager.list_models()
        self.model_combo.clear()
        for model in models:
            self.model_combo.addItem(model)

    def start_inference(self):
        model_name = self.model_combo.currentText()
        prompt = self.prompt_input.toPlainText()
        temp = float(self.temp_input.text())
        max_tokens = int(self.max_tokens_input.text())

        self.worker = InferenceWorker(model_name, prompt, temp, max_tokens)
        self.worker.update_output.connect(self.update_output)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.inference_done)
        self.worker.start()

    def update_output(self, text):
        self.output_display.append(text)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.status_label.setText(f"进度: {value}%")
        QApplication.processEvents()  # 强制刷新界面

    def inference_done(self):
        self.progress_bar.setValue(0)
        self.status_label.setText("等待操作...")

    def download_model(self):
        model_name = self.download_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "错误", "请输入模型名称。")
            return
        self.worker = DownloadWorker(model_name)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(lambda success, name: self.handle_download_result(success, name))
        self.worker.start()

    def handle_download_result(self, success, model_name):
        if success:
            self.load_models()
            QMessageBox.information(self, "成功", f"模型 {model_name} 下载完成！")
        else:
            QMessageBox.critical(self, "失败", f"模型 {model_name} 下载失败。")
        self.progress_bar.setValue(0)
        self.status_label.setText("等待操作...")

    def delete_model(self):
        model_name = self.model_combo.currentText()
        confirm = QMessageBox.question(self, "确认删除", f"确定要删除模型 {model_name}？",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            result = self.model_manager.delete_model(model_name)
            if result:
                self.load_models()
                QMessageBox.information(self, "删除成功", f"模型 {model_name} 已删除。")
            else:
                QMessageBox.critical(self, "删除失败", f"无法删除模型 {model_name}。")

    def test_token_speed(self):
        model_name = self.model_combo.currentText()
        prompt = (
            "The quick brown fox jumps over the lazy dog. "
            "Once upon a time in a land far, far away, there lived a brave knight and a wise wizard. "
            "They embarked on a journey to find the legendary treasure of Eldoria."
        )
        self.worker = InferenceWorker(model_name, prompt, 0.1, 500, speed_test=True)
        self.worker.update_output.connect(self.update_output)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.speed_result.connect(self.show_speed_result)
        self.worker.start()

    def show_speed_result(self, tokens, seconds):
        if tokens == 0 or seconds <= 0:
            QMessageBox.warning(self, "测试失败", "未能获取有效的 token 输出。")
            return
        speed = tokens / seconds
        msg = f"输出了 {tokens} 个 token，耗时 {seconds:.2f} 秒。\n平均速度：{speed:.2f} tokens/s"
        QMessageBox.information(self, "Token 速度测试结果", msg)


def excepthook(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    QMessageBox.critical(None, "致命错误", str(exc_value))
    sys.exit(1)


if __name__ == "__main__":
    import sys
    import traceback

    sys.excepthook = excepthook

    app = QApplication(sys.argv)
    window = OllamaApp()
    window.show()
    sys.exit(app.exec_())