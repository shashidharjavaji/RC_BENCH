import sys
import json
import fitz  # PyMuPDF
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, 
                            QListWidget, QLabel, QScrollArea, QComboBox,
                            QTextEdit, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
import os



class TextEditDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Selected Text")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)

        layout = QVBoxLayout(self)

        # Add instruction label
        instruction_label = QLabel("Edit the selected text below:")
        layout.addWidget(instruction_label)

        # Text editor
        self.text_edit = QTextEdit()
        self.text_edit.setText(text)
        layout.addWidget(self.text_edit)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def get_edited_text(self):
        return self.text_edit.toPlainText()

class PDFPageWidget(QWidget):
    textSelected = pyqtSignal(str)
    
    def __init__(self, page, zoom=1.5):
        super().__init__()
        self.page = page
        self.zoom = zoom
        self.pixmap = None
        self.start_pos = None
        self.current_pos = None
        self.accumulated_text = []  # Store multiple selections
        self.initialize_page()
        self.setMouseTracking(True)

    def initialize_page(self):
        """Initialize the page display"""
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pix = self.page.get_pixmap(matrix=matrix)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(img)
        self.setFixedSize(self.pixmap.size())

    def clear_selection(self):
        """Clear accumulated selections"""
        self.accumulated_text = []
        self.textSelected.emit("")  # Emit empty string to clear the text box

    def paintEvent(self, event):
        if not self.pixmap:
            return

        painter = QPainter(self)
        try:
            # Draw page
            painter.drawPixmap(0, 0, self.pixmap)

            # Draw selection rectangle
            if self.start_pos and self.current_pos:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 255, 0, 100))  # Semi-transparent yellow
                
                x = min(self.start_pos.x(), self.current_pos.x())
                y = min(self.start_pos.y(), self.current_pos.y())
                width = abs(self.current_pos.x() - self.start_pos.x())
                height = abs(self.current_pos.y() - self.start_pos.y())
                
                painter.drawRect(x, y, width, height)
        finally:
            painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.current_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.current_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_pos and self.current_pos:
            # Convert screen coordinates to PDF coordinates
            start_x = min(self.start_pos.x(), self.current_pos.x()) / self.zoom
            start_y = min(self.start_pos.y(), self.current_pos.y()) / self.zoom
            end_x = max(self.start_pos.x(), self.current_pos.x()) / self.zoom
            end_y = max(self.start_pos.y(), self.current_pos.y()) / self.zoom

            # Get text from the selected area
            rect = fitz.Rect(start_x, start_y, end_x, end_y)
            text = self.page.get_text("text", clip=rect)
            
            # Clean up the text
            text = self.clean_text(text)
            
            if text.strip():
                self.accumulated_text.append(text.strip())
                # Emit all accumulated text
                combined_text = ' '.join(self.accumulated_text)
                self.textSelected.emit(combined_text)

            self.start_pos = None
            self.current_pos = None
            self.update()

    def clean_text(self, text):
        """Clean up extracted text"""
        # Remove excessive newlines
        text = ' '.join(text.split('\n'))
        # Fix spacing issues
        text = ' '.join(text.split())
        # Remove any other unwanted characters or patterns
        text = text.replace('  ', ' ')  # Remove double spaces
        return text.strip()
class PDFViewer(QScrollArea):
    textSelected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.pages = []
        self.setWidget(self.container)
        self.setWidgetResizable(True)
        
        # Improved scrolling behavior
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.verticalScrollBar().setSingleStep(20)
        self.horizontalScrollBar().setSingleStep(20)

    def clear_selections(self):
        """Clear selections from all pages"""
        for page in self.pages:
            page.clear_selection()

    def load_pdf(self, doc):
        try:
            # Clear existing pages
            for page in self.pages:
                self.layout.removeWidget(page)
                page.deleteLater()
            self.pages.clear()

            # Add new pages
            for page_num in range(len(doc)):
                page_widget = PDFPageWidget(doc[page_num])
                page_widget.textSelected.connect(self.textSelected.emit)
                self.pages.append(page_widget)
                self.layout.addWidget(page_widget)

        except Exception as e:
            print(f"Error loading PDF: {str(e)}")

class PDFAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.claims = []
        self.claim_evidence_pairs = {}
        self.current_pdf_path = None
        self.doc = None
        self.selected_text = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PDF Annotation Tool')
        self.setGeometry(100, 100, 1400, 800)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel for PDF display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Controls panel
        controls_layout = QHBoxLayout()
        
        # Load button
        load_button = QPushButton('Load PDF')
        load_button.setMinimumWidth(100)
        load_button.clicked.connect(self.load_pdf)
        controls_layout.addWidget(load_button)
        controls_layout.addStretch()  # Add stretch to push the load button to the left
        
        # Add controls layout to left panel
        left_layout.addLayout(controls_layout)

        # PDF viewer
        self.pdf_viewer = PDFViewer()
        self.pdf_viewer.textSelected.connect(self.handle_text_selected)
        left_layout.addWidget(self.pdf_viewer)

        # Right panel for annotations
        right_panel = QWidget()
        right_panel.setMinimumWidth(400)
        right_panel.setMaximumWidth(500)
        right_layout = QVBoxLayout(right_panel)

        # Selection info
        selection_label_header = QLabel("Current Selection:")
        selection_label_header.setStyleSheet("QLabel { font-weight: bold; }")
        right_layout.addWidget(selection_label_header)

        self.selection_label = QLabel("Selected text will appear here")
        self.selection_label.setWordWrap(True)
        self.selection_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        self.selection_label.setMinimumHeight(150)
        self.selection_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        right_layout.addWidget(self.selection_label)

        # Selection control buttons
        selection_buttons_layout = QHBoxLayout()
        
        # Clear button
        self.clear_button = QPushButton('Clear Selection')
        self.clear_button.clicked.connect(self.clear_selection)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9999;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ff7777;
            }
        """)
        
        # Preview button
        self.preview_button = QPushButton('Preview & Edit')
        self.preview_button.clicked.connect(self.preview_selection)
        self.preview_button.setEnabled(False)
        self.preview_button.setStyleSheet("""
            QPushButton {
                background-color: #99ccff;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #77aaff;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        selection_buttons_layout.addWidget(self.clear_button)
        selection_buttons_layout.addWidget(self.preview_button)
        right_layout.addLayout(selection_buttons_layout)

        # Add claim/evidence buttons
        action_buttons_layout = QHBoxLayout()
        
        self.add_claim_button = QPushButton('Add as Claim')
        self.add_claim_button.clicked.connect(self.add_claim)
        self.add_claim_button.setEnabled(False)
        self.add_claim_button.setStyleSheet("""
            QPushButton {
                background-color: #99ff99;
                padding: 8px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #77ff77;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        self.add_evidence_button = QPushButton('Add as Evidence')
        self.add_evidence_button.clicked.connect(self.add_evidence)
        self.add_evidence_button.setEnabled(False)
        self.add_evidence_button.setStyleSheet("""
            QPushButton {
                background-color: #ffcc99;
                padding: 8px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ffaa77;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        action_buttons_layout.addWidget(self.add_claim_button)
        action_buttons_layout.addWidget(self.add_evidence_button)
        right_layout.addLayout(action_buttons_layout)

        # Claims section
        claims_label = QLabel('Claims:')
        claims_label.setStyleSheet("QLabel { font-weight: bold; margin-top: 10px; }")
        right_layout.addWidget(claims_label)
        
        self.claims_list = QListWidget()
        self.claims_list.itemClicked.connect(self.claim_selected)
        self.claims_list.setMinimumHeight(150)
        self.claims_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: white;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #e6f3ff;
            }
        """)
        right_layout.addWidget(self.claims_list)

        # Claim selector for evidence
        selector_label = QLabel('Select Claim for Evidence:')
        selector_label.setStyleSheet("QLabel { font-weight: bold; margin-top: 10px; }")
        right_layout.addWidget(selector_label)
        
        self.claim_selector = QComboBox()
        self.claim_selector.setMinimumWidth(200)
        self.claim_selector.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        right_layout.addWidget(self.claim_selector)

        # Evidence section
        evidence_label = QLabel('Evidence:')
        evidence_label.setStyleSheet("QLabel { font-weight: bold; margin-top: 10px; }")
        right_layout.addWidget(evidence_label)
        
        self.evidence_list = QListWidget()
        self.evidence_list.setMinimumHeight(150)
        self.evidence_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: white;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #e6f3ff;
            }
        """)
        right_layout.addWidget(self.evidence_list)

        # Save button
        save_button = QPushButton('Save Annotations')
        save_button.clicked.connect(self.save_annotations)
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 3px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        right_layout.addWidget(save_button)

        # Add panels to main layout
        layout.addWidget(left_panel, stretch=2)
        layout.addWidget(right_panel, stretch=1)

        # Status bar
        self.statusBar().showMessage('Ready')
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #f0f0f0;
                padding: 3px;
                border-top: 1px solid #ddd;
            }
        """)

    def clear_selection(self):
        """Clear all selections"""
        self.pdf_viewer.clear_selections()
        self.selected_text = ""
        self.selection_label.setText("Selected text will appear here")
        self.preview_button.setEnabled(False)
        self.add_claim_button.setEnabled(False)
        self.add_evidence_button.setEnabled(False)

    def load_pdf(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Open PDF file", "", "PDF files (*.pdf)")
            if file_name:
                if self.doc:
                    self.doc.close()
                self.current_pdf_path = file_name
                self.doc = fitz.open(file_name)
                self.pdf_viewer.load_pdf(self.doc)
                self.statusBar().showMessage(f'Loaded: {os.path.basename(file_name)}')
        except Exception as e:
            self.statusBar().showMessage(f'Error loading PDF: {str(e)}')

    def handle_text_selected(self, text):
        self.selected_text = text
        display_text = text[:200] + "..." if len(text) > 200 else text
        self.selection_label.setText(f"Selected: {display_text}")
        self.preview_button.setEnabled(True)
        self.add_claim_button.setEnabled(False)
        self.add_evidence_button.setEnabled(False)

    def preview_selection(self):
        if not self.selected_text:
            return

        dialog = TextEditDialog(self.selected_text, self)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_text = dialog.get_edited_text()
            self.add_claim_button.setEnabled(True)
            self.add_evidence_button.setEnabled(True)
            display_text = self.selected_text[:200] + "..." if len(self.selected_text) > 200 else self.selected_text
            self.selection_label.setText(f"Edited Text: {display_text}")
            self.statusBar().showMessage('Text edited successfully')
        else:
            self.selected_text = ""
            self.add_claim_button.setEnabled(False)
            self.add_evidence_button.setEnabled(False)
            self.selection_label.setText("Selected text will appear here")
            self.statusBar().showMessage('Text editing cancelled')

    def add_claim(self):
        if self.selected_text:
            self.claims.append(self.selected_text)
            self.claims_list.addItem(self.selected_text)
            self.claim_selector.addItem(self.selected_text)
            self.claim_evidence_pairs[self.selected_text] = []
            self.selected_text = ""
            self.selection_label.setText("Selected text will appear here")
            self.add_claim_button.setEnabled(False)
            self.add_evidence_button.setEnabled(False)
            self.preview_button.setEnabled(False)
            self.statusBar().showMessage('Claim added successfully')

    def add_evidence(self):
        if not self.selected_text:
            return

        selected_claim = self.claim_selector.currentText()
        if not selected_claim:
            self.statusBar().showMessage('Please select a claim first')
            return

        self.claim_evidence_pairs[selected_claim].append(self.selected_text)
        self.evidence_list.addItem(self.selected_text)
        self.selected_text = ""
        self.selection_label.setText("Selected text will appear here")
        self.add_claim_button.setEnabled(False)
        self.add_evidence_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.statusBar().showMessage('Evidence added successfully')

    def claim_selected(self, item):
        claim_text = item.text()
        self.evidence_list.clear()
        for evidence in self.claim_evidence_pairs[claim_text]:
            self.evidence_list.addItem(evidence)
        self.claim_selector.setCurrentText(claim_text)
        self.statusBar().showMessage(f'Showing evidence for selected claim')

    def save_annotations(self):
        if not self.current_pdf_path or not self.claim_evidence_pairs:
            self.statusBar().showMessage('Nothing to save')
            return

        try:
            base_name = os.path.splitext(self.current_pdf_path)[0]
            output_file = f"{base_name}_human_annotated.json"
            
            annotations = []
            for claim, evidences in self.claim_evidence_pairs.items():
                annotations.append({
                    "claim": claim,
                    "evidences": evidences
                })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_pdf": self.current_pdf_path,
                    "annotations": annotations
                }, f, indent=4, ensure_ascii=False)
            
            self.statusBar().showMessage(f'Annotations saved to: {os.path.basename(output_file)}')
        except Exception as e:
            self.statusBar().showMessage(f'Error saving annotations: {str(e)}')

    def closeEvent(self, event):
        if self.doc:
            self.doc.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    ex = PDFAnnotationTool()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
