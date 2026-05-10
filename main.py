import sys
import os
import json
import keyboard
import base64
import re
import time
import requests
import tempfile
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QComboBox, QGraphicsOpacityEffect, QFrame
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PyQt6.QtCore import (
    Qt, QRect, QPoint, pyqtSignal, pyqtSlot, QThread, QTimer,
    QPropertyAnimation, QEasingCurve
)

# ─── Config persistence ───────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(data: dict):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# ─── API Worker ───────────────────────────────────────────────────────────────
class APIWorker(QThread):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, prompt, image_path, api_key, provider, model):
        super().__init__()
        self.prompt     = prompt
        self.image_path = image_path
        self.api_key    = api_key
        self.provider   = provider
        self.model      = model

    # Gemini 2.5 models: use thinkingBudget (0 = off, -1 = dynamic)
    GEMINI_25_MODELS = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"}
    # Gemini 3.x models: use thinkingLevel ("minimal"/"low"/"medium"/"high")
    # Note: thinking cannot be fully disabled on 3.1 Pro
    GEMINI_3X_MODELS = {"gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview"}

    def _extract_gemini_text(self, data):
        """Extract text from Gemini response, handling thinking model multi-part responses.
        Thinking models return parts with 'thought':True for internal reasoning
        and regular parts for the actual answer. We only want the answer parts."""
        candidates = data.get("candidates", [])
        if not candidates:
            return "No response generated."

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return "No text parsed."

        # Collect all non-thought text parts
        answer_parts = []
        for part in parts:
            # Skip thought/reasoning parts (thinking models)
            if part.get("thought", False):
                continue
            text = part.get("text", "")
            if text:
                answer_parts.append(text)

        if answer_parts:
            return "\n".join(answer_parts)

        # Fallback: if all parts were thoughts, just take the last part's text
        for part in reversed(parts):
            text = part.get("text", "")
            if text:
                return text

        return "No text parsed."

    def _gemini_request(self, headers, b64_img):
        """Make a Gemini API request with automatic retry on 429."""
        url = (f"https://generativelanguage.googleapis.com/v1beta"
               f"/models/{self.model}:generateContent")
        headers["X-goog-api-key"] = self.api_key
        formatted_prompt = (
            f"{self.prompt}\n\n"
            "Please respond in a clear, structured format using short paragraphs "
            "or bullet points for easy reading."
        )
        payload = {
            "contents": [{"parts": [
                {"text": formatted_prompt},
                {"inline_data": {"mime_type": "image/png", "data": b64_img}}
            ]}],
        }

        # Configure thinking per model family (official Google docs):
        # - Gemini 2.5: thinkingBudget (0 = disable thinking)
        # - Gemini 3.x: thinkingLevel ("minimal" = near-off, can't fully disable 3.1 Pro)
        if self.model in self.GEMINI_25_MODELS:
            payload["generationConfig"] = {
                "thinkingConfig": {"thinkingBudget": 0}
            }
        elif self.model in self.GEMINI_3X_MODELS:
            payload["generationConfig"] = {
                "thinkingConfig": {"thinkingLevel": "minimal"}
            }

        # Pro/preview models may need longer timeouts
        timeout = 120 if "pro" in self.model else 90

        max_retries = 2
        for attempt in range(max_retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            data = resp.json()

            if resp.status_code == 200:
                text = self._extract_gemini_text(data)
                self.finished.emit(text)
                return

            if resp.status_code == 429 and attempt < max_retries - 1:
                err_msg = data.get("error", {}).get("message", "")
                # Check if quota limit is literally 0 (free tier exhausted)
                if "limit: 0" in err_msg:
                    self.error.emit(
                        f"⚠ Free tier quota exhausted for '{self.model}'.\n"
                        "Your Gemini API key has hit the daily free-tier limit "
                        "for this specific model.\n\n"
                        "Fix options:\n"
                        "• Try a different model (e.g. gemini-2.5-flash)\n"
                        "• Wait until tomorrow for the quota to reset\n"
                        "• Enable billing at console.cloud.google.com\n"
                        "• Generate a new API key at aistudio.google.com"
                    )
                    return
                # Transient rate limit — extract retry delay and wait
                retry_match = re.search(r'retry in ([\d.]+)s', err_msg, re.IGNORECASE)
                wait_secs = float(retry_match.group(1)) if retry_match else 10
                wait_secs = min(wait_secs, 30)  # cap at 30s
                time.sleep(wait_secs)
                continue

            # Non-retryable error
            err = data.get("error", {})
            self.error.emit(f"[{resp.status_code}] {err.get('message', str(data))}")
            return

    def run(self):
        try:
            if not self.api_key:
                self.error.emit("API Key is missing. Enter it in the Control Panel.")
                return
            with open(self.image_path, "rb") as f:
                img_data = f.read()
            b64_img = base64.b64encode(img_data).decode("utf-8")
            headers = {"Content-Type": "application/json"}

            # ── Gemini ──
            if self.provider == "Google Gemini":
                self._gemini_request(headers, b64_img)

            # ── OpenAI ──
            elif self.provider == "OpenAI":
                url = "https://api.openai.com/v1/chat/completions"
                headers["Authorization"] = f"Bearer {self.api_key}"
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]}],
                    "max_tokens": 1200
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=90)
                data = resp.json()
                if resp.status_code == 200:
                    # Try new interactions format first
                    text = ""
                    if "output" in data:
                        try:
                            text = data["output"][0]["content"][0].get("text", "")
                        except (KeyError, IndexError):
                            pass
                    
                    # Fallback to standard chat completions
                    if not text:
                        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    self.finished.emit(text if text else "No text found in response.")
                else:
                    self.error.emit(f"[{resp.status_code}] {data.get('error', {}).get('message', str(data))}")

            # ── Anthropic ──
            elif self.provider == "Anthropic":
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "max_tokens": 1024,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64_img
                                }
                            },
                            {
                                "type": "text",
                                "text": self.prompt
                            }
                        ]
                    }]
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=90)
                data = resp.json()
                if resp.status_code == 200:
                    content_list = data.get("content", [])
                    text = ""
                    for item in content_list:
                        if item.get("type") == "text":
                            text += item.get("text", "")
                    self.finished.emit(text if text else "No text response.")
                else:
                    err = data.get("error", {})
                    self.error.emit(f"[{resp.status_code}] {err.get('message', str(data))}")

        except Exception as e:
            self.error.emit(str(e))

# ─── Shortcut recorder ────────────────────────────────────────────────────────
class ShortcutLineEdit(QLineEdit):
    def __init__(self, default_text=""):
        super().__init__(default_text)
        self.setReadOnly(True)
        self.setPlaceholderText("Click here, then press your shortcut keys…")
        self._style()

    def _style(self):
        self.setStyleSheet(
            "background:#2b2b2b; color:#fff; border:1px solid #555; "
            "padding:6px; font-size:14px; border-radius:5px;"
        )

    def mousePressEvent(self, event):
        self.setText("⌨  Press key combo…")
        from threading import Thread
        Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        hotkey = keyboard.read_hotkey(suppress=False)
        from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(self, "setText",
                                 Qt.ConnectionType.QueuedConnection, Q_ARG(str, hotkey))

# ─── Overlay (screenshot + AI panel) ─────────────────────────────────────────
class Overlay(QWidget):
    closed_signal = pyqtSignal()

    def __init__(self, pixmap, geometry, api_key="", provider="Google Gemini",
                 model="gemini-flash-latest"):
        super().__init__()
        self.api_key  = api_key
        self.provider = provider
        self.model    = model

        self.full_response  = ""
        self.current_typed  = ""
        self.type_index     = 0
        self.type_timer     = QTimer(self)
        self.type_timer.timeout.connect(self._type_next_char)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.bg_pixmap      = pixmap
        self.start_point    = QPoint()
        self.end_point      = QPoint()
        self.is_drawing     = False
        self.selection_rect = QRect()
        self.setGeometry(geometry)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # ── Floating action panel ──────────────────────────────────────────
        self.panel = QWidget(self)
        self.panel.setFixedSize(520, 370)
        self.panel.setStyleSheet(
            "background:#1a1a1a; border:1px solid #3a3a3a; border-radius:10px;"
        )

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)

        # Header row: label + close button
        header = QHBoxLayout()
        lbl = QLabel("✦ AI Copilot")
        lbl.setStyleSheet("color:#aaa; font-size:13px; border:none; background:transparent;")
        header.addWidget(lbl)
        header.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet(
            "QPushButton{background:#333;color:#bbb;border:none;border-radius:14px;"
            "font-size:14px;font-weight:bold;}"
            "QPushButton:hover{background:#c0392b;color:#fff;}"
        )
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close_overlay)
        header.addWidget(close_btn)
        panel_layout.addLayout(header)

        # Prompt field
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Describe what to analyse in the screenshot…")
        self.text_edit.setMaximumHeight(72)
        self.text_edit.setStyleSheet(
            "color:#fff; background:#252525; border:1px solid #555; "
            "font-family:'Segoe UI',Arial; font-size:15px; border-radius:6px; padding:5px;"
        )
        panel_layout.addWidget(self.text_edit)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        # Capture again button
        self.recapture_btn = QPushButton("⟳  Re-capture")
        self.recapture_btn.setStyleSheet(
            "QPushButton{background:#333;color:#ccc;border:none;border-radius:6px;"
            "font-size:14px;padding:7px 14px;}"
            "QPushButton:hover{background:#444;}"
        )
        self.recapture_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recapture_btn.clicked.connect(self._recapture)
        btn_row.addWidget(self.recapture_btn)

        btn_row.addStretch()

        self.submit_btn = QPushButton("  Ask AI")
        self.submit_btn.setStyleSheet(
            "QPushButton{background:#0078D7;color:#fff;border:none;border-radius:6px;"
            "font-size:15px;font-weight:bold;padding:7px 20px;}"
            "QPushButton:hover{background:#005fa3;}"
            "QPushButton:disabled{background:#333;color:#666;}"
        )
        self.submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.submit_btn.clicked.connect(self.submit_to_ai)
        btn_row.addWidget(self.submit_btn)

        panel_layout.addLayout(btn_row)

        # Response area
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setStyleSheet(
            "color:#e8e8e8; background:#222; border:1px solid #333; "
            "font-family:'Segoe UI',Arial; font-size:15px; "
            "border-radius:6px; padding:8px; line-height:1.5;"
        )
        panel_layout.addWidget(self.result_box)

        self.panel.hide()

    # ── Drawing ────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close_overlay()

    def mousePressEvent(self, event):
        # Single click outside popup = close popup
        if self.panel.isVisible():
            if not self.panel.geometry().contains(event.pos()):
                self.close_overlay()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point   = self.start_point
            self.is_drawing  = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing     = False
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            if self.selection_rect.width() > 10 and self.selection_rect.height() > 10:
                self._show_panel()
            self.update()

    def _show_panel(self):
        w, h = 520, 370
        x = self.selection_rect.right() - w + 10
        y = self.selection_rect.bottom() + 12
        if y + h > self.height():
            y = self.selection_rect.top() - h - 12
        if x < 0:
            x = 10
        self.panel.move(x, y)
        self.panel.show()
        self.text_edit.setFocus()

    def _recapture(self):
        """Hide panel and let user draw a new selection."""
        self.panel.hide()
        self.selection_rect = QRect()
        self.result_box.clear()
        self.is_drawing = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.bg_pixmap)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        rect = (QRect(self.start_point, self.end_point).normalized()
                if self.is_drawing else self.selection_rect)

        if not rect.isNull():
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.drawPixmap(rect.topLeft(), self.bg_pixmap.copy(rect))
            pen = QPen(QColor("#0078D7"), 2)
            painter.setPen(pen)
            painter.drawRect(rect)

    # ── AI submission ──────────────────────────────────────────────────────
    def submit_to_ai(self):
        prompt = self.text_edit.toPlainText().strip()
        if not prompt or self.selection_rect.isNull():
            return
        crop     = self.bg_pixmap.copy(self.selection_rect)
        path     = os.path.join(tempfile.gettempdir(), "ai_crop.png")
        crop.save(path)
        QApplication.clipboard().setPixmap(crop)

        self.result_box.setPlainText("Loading…")
        self.submit_btn.setEnabled(False)

        self.worker = APIWorker(prompt, path, self.api_key, self.provider, self.model)
        self.worker.finished.connect(self._on_response)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_response(self, text):
        self.full_response = text
        self.current_typed = ""
        self.type_index    = 0
        self.result_box.clear()
        self.type_timer.start(8)

    def _on_error(self, msg):
        self.result_box.setPlainText(f"Error: {msg}")
        self.submit_btn.setEnabled(True)

    def _type_next_char(self):
        if self.type_index < len(self.full_response):
            self.current_typed += self.full_response[self.type_index]
            self.result_box.setPlainText(self.current_typed)
            # Auto-scroll to bottom
            sb = self.result_box.verticalScrollBar()
            sb.setValue(sb.maximum())
            self.type_index += 1
        else:
            self.type_timer.stop()
            self.submit_btn.setEnabled(True)

    def close_overlay(self):
        self.type_timer.stop()
        self.close()
        self.closed_signal.emit()

# ─── Assistant Overlay (Result Window) ────────────────────────────────────────
class AssistantOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Load saved size
        cfg = load_config()
        sizes = cfg.get("window_sizes", {})
        a_size = sizes.get("AssistantOverlay", [600, 450])
        self.resize(a_size[0], a_size[1])

        # ── UI ────────────────────────────────────────────────────────────
        self.container = QFrame(self)
        self.container.setObjectName("ResultContainer")
        self.container.setStyleSheet("""
            #ResultContainer {
                background: rgba(25, 25, 25, 235);
                border: 1px solid #444;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(15, 10, 15, 15)

        # Title / Handle (simplified resize/move area)
        title_bar = QHBoxLayout()
        title_lbl = QLabel("AI Analysis")
        title_lbl.setStyleSheet("color:#888; font-weight:bold; font-size:12px; text-transform:uppercase;")
        title_bar.addWidget(title_lbl)
        
        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("background:transparent; color:#888; font-size:20px; border:none;")
        close_btn.clicked.connect(self.close)
        title_bar.addStretch()
        title_bar.addWidget(close_btn)
        layout.addLayout(title_bar)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("""
            QTextEdit {
                background: transparent;
                color: #e0e0e0;
                border: none;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(self.text_area)

    def resizeEvent(self, event):
        """Preserve result window size."""
        super().resizeEvent(event)
        self.container.setGeometry(self.rect())
        # Save size on resize
        cfg = load_config()
        if "window_sizes" not in cfg: cfg["window_sizes"] = {}
        cfg["window_sizes"]["AssistantOverlay"] = [self.width(), self.height()]
        save_config(cfg)

# ─── Control Panel ────────────────────────────────────────────────────────────
class ConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Copilot")
        self.setMinimumSize(480, 460)
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#1a1a1a; color:#e0e0e0; "
            "font-family:'Segoe UI',Arial; font-size:15px;}"
        )
        self.overlay   = None
        self.is_active = False

        cfg = load_config()
        self.saved_keys = cfg.get("api_keys", {
            "Google Gemini": cfg.get("api_key", ""),
            "OpenAI": "",
            "Anthropic": ""
        })

        # Load saved size
        sizes = cfg.get("window_sizes", {})
        c_size = sizes.get("ConfigWindow", [480, 460])
        self.resize(c_size[0], c_size[1])

        # ── Layout ────────────────────────────────────────────────────────
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        self.setCentralWidget(root)

        # Status row
        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("font-size:22px; color:#e74c3c;")
        self.status_lbl = QLabel("Service inactive")
        self.status_lbl.setStyleSheet("color:#888; font-size:13px;")
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_lbl)
        status_row.addStretch()
        layout.addLayout(status_row)

        # Provider
        layout.addWidget(self._lbl("Provider"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Google Gemini", "OpenAI", "Anthropic"])
        self.provider_combo.setStyleSheet(self._combo_style())
        saved_provider = cfg.get("provider", "Google Gemini")
        idx = self.provider_combo.findText(saved_provider)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        layout.addWidget(self.provider_combo)

        # Model
        layout.addWidget(self._lbl("Model"))
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(self._combo_style())
        self._update_models(self.provider_combo.currentText(), restore=cfg.get("model"))
        layout.addWidget(self.model_combo)

        # API key
        layout.addWidget(self._lbl("API Key"))
        key_row = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Paste your API key here…")
        self.api_key_input.setStyleSheet(self._input_style())
        
        current_provider = self.provider_combo.currentText()
        self.api_key_input.setText(self.saved_keys.get(current_provider, ""))
        key_row.addWidget(self.api_key_input)

        self.show_key_btn = QPushButton()
        self.show_key_btn.setFixedWidth(40)
        self.show_key_btn.setFixedHeight(36)
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_key_btn.clicked.connect(self._toggle_key_visibility)
        self._update_eye_icon(False)
        key_row.addWidget(self.show_key_btn)
        layout.addLayout(key_row)

        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)

        # Shortcut
        layout.addWidget(self._lbl("Trigger Shortcut  (click field then press keys)"))
        self.shortcut_input = ShortcutLineEdit(cfg.get("shortcut", "ctrl+shift+a"))
        layout.addWidget(self.shortcut_input)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.capture_btn = QPushButton("📷  Capture Now")
        self.capture_btn.setStyleSheet(
            "QPushButton{background:#333; color:#aaa; border:1px solid #444; border-radius:8px;"
            "font-family:'Segoe UI Semibold', 'Segoe UI', Arial; font-size:15px; padding:8px 15px; min-height:44px;}"
            "QPushButton:enabled{background:#1e8449; color:#fff; border:none;}"
            "QPushButton:hover:enabled{background:#196f3d;}"
        )
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self._manual_capture)
        btn_row.addWidget(self.capture_btn)

        self.toggle_btn = QPushButton("▶  Start Service")
        self.toggle_btn.setStyleSheet(
            "QPushButton{background:#0078D7; color:#fff; border:none; border-radius:8px;"
            "font-family:'Segoe UI Semibold', 'Segoe UI', Arial; font-size:15px; padding:8px 15px; min-height:44px;}"
            "QPushButton:hover{background:#005fa3;}"
        )
        self.toggle_btn.clicked.connect(self.toggle_service)
        btn_row.addWidget(self.toggle_btn)

        layout.addLayout(btn_row)
        
        # Social Footer
        layout.addSpacing(30)
        footer_wrap = QWidget()
        footer_layout = QHBoxLayout(footer_wrap)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        
        follow_txt = QLabel("FOLLOW US:")
        follow_txt.setStyleSheet("color:#555; font-size:11px; font-weight:bold; letter-spacing:1px;")
        footer_layout.addWidget(follow_txt)
        footer_layout.addSpacing(10)

        # YouTube Button
        self.yt_btn = QPushButton()
        self._set_social_icon(self.yt_btn, "youtube", "https://www.youtube.com/@GenesisDigitalSolutions")
        footer_layout.addWidget(self.yt_btn)
        
        footer_layout.addSpacing(10)

        # Instagram Button
        self.ig_btn = QPushButton()
        self._set_social_icon(self.ig_btn, "instagram", "https://www.instagram.com/genesis.digitals/")
        footer_layout.addWidget(self.ig_btn)

        footer_layout.addSpacing(10)

        # GitHub Button
        self.gh_btn = QPushButton()
        self._set_social_icon(self.gh_btn, "github", "https://github.com/Genesis-Digital-Lab")
        footer_layout.addWidget(self.gh_btn)

        footer_layout.addStretch()
        layout.addWidget(footer_wrap)
        layout.addStretch()

        # Save config on any change
        self.api_key_input.textChanged.connect(self._on_key_edited)
        self.provider_combo.currentTextChanged.connect(lambda _: self._save())
        self.model_combo.currentTextChanged.connect(lambda _: self._save())
        self.shortcut_input.textChanged.connect(lambda _: self._save())

    def _on_provider_changed(self, provider):
        """Handle swapping API keys and updating models when provider changes."""
        self._update_models(provider)
        self.api_key_input.setText(self.saved_keys.get(provider, ""))
        self._save()

    def _on_key_edited(self, text):
        """Keep the internal key map updated as the user types."""
        provider = self.provider_combo.currentText()
        self.saved_keys[provider] = text
        self._save()

    # ── Helpers ────────────────────────────────────────────────────────────
    def _lbl(self, text):
        l = QLabel(text)
        l.setStyleSheet("color:#aaa; font-size:13px; margin-top:4px;")
        return l

    def _combo_style(self):
        return (
            "QComboBox{background:#2b2b2b;color:#fff;border:1px solid #555;"
            "border-radius:5px;padding:6px;font-size:15px;}"
            "QComboBox QAbstractItemView{background:#2b2b2b;color:#fff;selection-background-color:#0078D7;}"
        )

    def _input_style(self):
        return (
            "background:#2b2b2b;color:#fff;border:1px solid #555;"
            "border-radius:5px;padding:6px;font-size:15px;"
        )

    def _toggle_key_visibility(self, checked):
        mode = QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        self.api_key_input.setEchoMode(mode)
        self._update_eye_icon(checked)

    def _update_eye_icon(self, visible):
        """Update the eye icon button with an SVG mimicking FontAwesome."""
        icon_color = "#ffffff" if visible else "#888888"
        # Simple SVG paths for Eye and Eye-Slash
        if visible: # Eye
            svg = f'<svg viewBox="0 0 576 512" width="20" height="20"><path fill="{icon_color}" d="M572.52 266.52c-29.35 44.53-125.13 173.48-284.52 173.48s-255.17-128.95-284.52-173.48c-10.67-16.19-10.67-41.25 0-57.44C32.83 164.55 128.61 35.59 288 35.59s255.17 128.95 284.52 173.48c10.67 16.19 10.67 41.25 0 57.44zM288 128a128 128 0 1 0 128 128 128 128 0 0 0-128-128zm0 192a64 64 0 1 1 64-64 64 64 0 0 1-64 64z"/></svg>'
        else: # Eye Slash
            svg = f'<svg viewBox="0 0 640 512" width="20" height="20"><path fill="{icon_color}" d="M320 400c-75.85 0-137.25-58.71-142.9-133.11L72.2 185.82c-13.79 17.3-26.48 35.59-36.72 55.59a32.35 32.35 0 0 0 0 29.19c31.51 57.89 121 163.4 284.52 163.4 13.52 0 27.67-1.07 41-2.92L320 400zm283.48-111.41C571.97 325.29 482.48 430.8 318.96 430.8a287.41 287.41 0 0 1-41-2.92L312 393.1l-10-38.41c-72.2-4.14-130.33-61.9-136-133.87l-64.21-36.69c-10.67 16.19-20.9 33.72-28.79 52.48a32.35 32.35 0 0 0 0 29.19c31.51 57.89 121 163.4 284.52 163.4 13.52 0 27.67-1.07 41-2.92l34.42-19.67c48.66-27.81 83.33-67.64 104.53-107.82-5.74-10.9-12.01-21.43-18.74-31.53zM608 448L495.2 335.2c41.33-31.42 75.31-69.31 100.28-111.41a32.35 32.35 0 0 0 0-29.19C564 136.71 474.52 31.2 311 31.2c-56.12 0-108.62 12.39-153.81 33.51L48 16zm-313 162.2a128.2 128.2 0 0 1-127.8-128c0-3.13.11-6.24.32-9.33L294 384l1-211.8z"/></svg>'
        
        from PyQt6.QtGui import QIcon, QPixmap
        from PyQt6.QtCore import QByteArray
        pixmap = QPixmap()
        pixmap.loadFromData(QByteArray(svg.encode()))
        
        self.show_key_btn.setIcon(QIcon(pixmap))
        self.show_key_btn.setStyleSheet(
            "QPushButton { background:#252525; border:1px solid #444; border-radius:5px; padding:4px; }"
            "QPushButton:hover { background:#333; border-color:#666; }"
        )

    def _set_social_icon(self, btn, type, url):
        """Set up a high-performance SVG social icon button."""
        if type == "youtube":
            svg = '<svg viewBox="0 0 576 512"><path fill="#ff0000" d="M549.655 124.083c-6.281-23.65-24.787-42.276-48.284-48.597C458.781 64 288 64 288 64S117.22 64 74.629 75.486c-23.497 6.322-42.003 24.947-48.284 48.597-11.412 42.867-11.412 132.305-11.412 132.305s0 89.438 11.412 132.305c6.281 23.65 24.787 41.5 48.284 47.821C117.22 448 288 448 288 448s170.781 0 213.371-11.486c23.497-6.321 42.003-24.171 48.284-47.821 11.412-42.867 11.412-132.305 11.412-132.305s0-89.438-11.412-132.305zm-317.51 213.508V175.185l142.739 81.205-142.739 81.201z"/></svg>'
        elif type == "instagram":
            svg = '<svg viewBox="0 0 448 512"><path fill="#E1306C" d="M224.1 141c-63.6 0-114.9 51.3-114.9 114.9s51.3 114.9 114.9 114.9S339 319.5 339 255.9 287.7 141 224.1 141zm0 189.6c-41.1 0-74.7-33.5-74.7-74.7s33.5-74.7 74.7-74.7 74.7 33.5 74.7 74.7-33.6 74.7-74.7 74.7zm146.4-194.3c0 14.9-12 26.8-26.8 26.8-14.9 0-26.8-12-26.8-26.8s12-26.8 26.8-26.8 26.8 12 26.8 26.8zm76.1 27.2c-1.7-35.9-9.9-67.7-36.2-93.9-26.2-26.2-58-34.4-93.9-36.2-37-2.1-147.9-2.1-184.9 0-35.8 1.7-67.6 9.9-93.9 36.1s-34.4 58-36.2 93.9c-2.1 37-2.1 147.9 0 184.9 1.7 35.9 9.9 67.7 36.2 93.9s58 34.4 93.9 36.2c37 2.1 147.9 2.1 184.9 0 35.9-1.7 67.7-9.9 93.9-36.2 26.2-26.2 34.4-58 36.2-93.9 2.1-37 2.1-147.8 0-184.8zM398.8 388c-7.8 19.6-22.9 34.7-42.6 42.6-29.5 11.7-99.5 9-132.1 9s-102.7 2.6-132.1-9c-19.6-7.8-34.7-22.9-42.6-42.6-11.7-29.5-9-99.5-9-132.1s-2.6-102.7 9-132.1c7.8-19.6 22.9-34.7 42.6-42.6 29.5-11.7 99.5-9 132.1-9s102.7-2.6 132.1 9c19.6 7.8 34.7 22.9 42.6 42.6 11.7 29.5 9 99.5 9 132.1s2.7 102.7-9 132.1z"/></svg>'
        else: # GitHub
            svg = '<svg viewBox="0 0 496 512"><path fill="#ffffff" d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg>'

        from PyQt6.QtGui import QIcon, QPixmap
        from PyQt6.QtCore import QByteArray, QUrl
        from PyQt6.QtGui import QDesktopServices
        pixmap = QPixmap()
        pixmap.loadFromData(QByteArray(svg.encode()))
        btn.setIcon(QIcon(pixmap))
        btn.setIconSize(btn.sizeHint())
        btn.setFixedSize(32, 32)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("QPushButton{background:transparent; border:none;} QPushButton:hover{background:#333; border-radius:16px;}")
        btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))

    def resizeEvent(self, event):
        """Preserve window size on resize."""
        super().resizeEvent(event)
        self._save()

    def _update_models(self, provider, restore=None):
        maps = {
            "Google Gemini": ["gemini-2.5-flash", "gemini-2.5-pro",
                              "gemini-2.5-flash-lite", "gemini-3.1-flash-lite"],
            "OpenAI":        ["gpt-4o", "gpt-4o-mini"],
            "Anthropic":     ["claude-sonnet-4-6", "claude-opus-4-6", 
                              "claude-sonnet-4-5-20250929", "claude-opus-4-20250514",
                              "claude-haiku-4-5-20251001"],
        }
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(maps.get(provider, []))
        if restore:
            idx = self.model_combo.findText(restore)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _save(self):
        cfg = load_config()
        sizes = cfg.get("window_sizes", {})
        sizes["ConfigWindow"] = [self.width(), self.height()]
        
        save_config({
            "provider": self.provider_combo.currentText(),
            "model":    self.model_combo.currentText(),
            "api_keys": self.saved_keys,
            "shortcut": self.shortcut_input.text(),
            "window_sizes": sizes
        })

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.showMinimized()

    # ── Service toggle ─────────────────────────────────────────────────────
    def toggle_service(self):
        if self.is_active:
            keyboard.unhook_all_hotkeys()
            self.is_active = False
            self.status_dot.setStyleSheet("font-size:22px; color:#e74c3c;")
            self.status_lbl.setText("Service inactive")
            self.toggle_btn.setText("▶  Start Service")
            self.capture_btn.setEnabled(False)
        else:
            shortcut = self.shortcut_input.text()
            try:
                keyboard.add_hotkey(shortcut, self.trigger_screenshot, suppress=True)
                self.is_active = True
                self.status_dot.setStyleSheet("font-size:22px; color:#2ecc71;")
                self.status_lbl.setText(f"Active — {shortcut}")
                self.toggle_btn.setText("⏹  Stop Service")
                self.capture_btn.setEnabled(True)
                self._save()
            except Exception as e:
                self.status_lbl.setText(f"Error: {e}")

    def trigger_screenshot(self):
        from PyQt6.QtCore import QMetaObject, Qt
        try:
            QMetaObject.invokeMethod(self, "_execute_gui_capture",
                                     Qt.ConnectionType.QueuedConnection)
        except Exception:
            pass

    def _manual_capture(self):
        self._execute_gui_capture()

    @pyqtSlot()
    def _execute_gui_capture(self):
        import mss
        with mss.MSS() as sct:
            mon = sct.monitors[0]
            sct_img = sct.grab(mon)
            img_qt  = QImage(sct_img.rgb, sct_img.width, sct_img.height,
                             sct_img.width * 3, QImage.Format.Format_RGB888)
            pixmap  = QPixmap.fromImage(img_qt)
            vr      = QRect(mon["left"], mon["top"], mon["width"], mon["height"])

        if self.overlay is not None:
            self.overlay.close()

        self.overlay = Overlay(
            pixmap, vr,
            self.api_key_input.text(),
            self.provider_combo.currentText(),
            self.model_combo.currentText()
        )
        self.overlay.closed_signal.connect(self.on_overlay_closed)
        self.overlay.show()

    def on_overlay_closed(self):
        self.overlay = None

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_ENABLE_HIGHDPI_SCALING"]  = "0"
    os.environ["QT_SCALE_FACTOR"]            = "1"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = ConfigWindow()
    window.show()
    sys.exit(app.exec())
