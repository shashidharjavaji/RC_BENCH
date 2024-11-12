import sys
import json
import fitz  # PyMuPDF
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, 
                            QListWidget, QLabel, QScrollArea, QComboBox,
                            QTextEdit, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QKeySequence
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

class ClaimWidget(QWidget):
    def __init__(self, claim_text, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        # Claim header
        claim_header = QLabel("Claim:")
        claim_header.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        layout.addWidget(claim_header)

        # Claim text
        claim_label = QLabel(claim_text)
        claim_label.setWordWrap(True)
        claim_label.setStyleSheet("""
            QLabel {
                background-color: #e8f5e9;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #c8e6c9;
            }
        """)
        layout.addWidget(claim_label)

        # Evidence header
        evidence_header = QLabel("Evidence:")
        evidence_header.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 5px;
                border-radius: 3px;
                margin-top: 5px;
            }
        """)
        layout.addWidget(evidence_header)

        # Evidence list
        self.evidence_list = QListWidget()
        self.evidence_list.setMinimumHeight(100)
        self.evidence_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #fff3e0;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ffe0b2;
            }
            QListWidget::item:last {
                border-bottom: none;
            }
        """)
        layout.addWidget(self.evidence_list)

        self.setStyleSheet("""
            ClaimWidget {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin: 5px;
            }
        """)

    def add_evidence(self, evidence_text):
        self.evidence_list.addItem(evidence_text)

class PDFPageWidget(QWidget):
    textSelected = pyqtSignal(str)

    def __init__(self, page, zoom=1.5):
        super().__init__()
        self.page = page
        self.min_zoom = 0.5
        self.max_zoom = 5.0
        self.zoom = min(max(zoom, self.min_zoom), self.max_zoom)
        self.pixmap = None
        self.start_pos = None
        self.current_pos = None
        self.accumulated_text = []  # Store multiple selections
        self.initialize_page()
        self.setMouseTracking(True)

    def initialize_page(self):
        """Initialize the page display"""
        try:
            matrix = fitz.Matrix(self.zoom, self.zoom)
            pix = self.page.get_pixmap(matrix=matrix)
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(img)
            self.setFixedSize(self.pixmap.size())
        except Exception as e:
            print(f"Error initializing page: {str(e)}")

    def set_zoom(self, zoom):
        """Update zoom level and refresh page"""
        self.zoom = min(max(zoom, self.min_zoom), self.max_zoom)
        self.initialize_page()
        self.update()

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

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming when Ctrl is pressed"""
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.parent().parent().zoom_in()  # Zoom in
            else:
                self.parent().parent().zoom_out()  # Zoom out
            event.accept()
        else:
            event.ignore()  # Let the scroll area handle normal scrolling
class PDFViewer(QScrollArea):
    textSelected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.pages = []
        self.current_zoom = 1.5
        self.setWidget(self.container)
        self.setWidgetResizable(True)
        
        # Improved scrolling behavior
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.verticalScrollBar().setSingleStep(20)
        self.horizontalScrollBar().setSingleStep(20)

        # Remove margins for better display
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

    def set_zoom(self, zoom):
        """Update zoom level for all pages"""
        self.current_zoom = zoom
        for page in self.pages:
            page.set_zoom(zoom)

    def zoom_in(self):
        """Increase zoom level"""
        self.set_zoom(self.current_zoom * 1.2)

    def zoom_out(self):
        """Decrease zoom level"""
        self.set_zoom(self.current_zoom / 1.2)

    def fit_to_width(self):
        """Fit PDF to window width"""
        if not self.pages:
            return
        available_width = self.viewport().width() - 20  # Account for scrollbar
        page_width = self.pages[0].pixmap.width() / self.current_zoom
        new_zoom = available_width / page_width
        self.set_zoom(new_zoom)

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
                page_widget = PDFPageWidget(doc[page_num], self.current_zoom)
                page_widget.textSelected.connect(self.handle_text_selected)
                self.pages.append(page_widget)
                self.layout.addWidget(page_widget)

            # Add stretch at the end
            self.layout.addStretch()
            
            # Fit to width after loading
            QApplication.processEvents()
            self.fit_to_width()
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")

    def handle_text_selected(self, text):
        self.textSelected.emit(text)

class PDFAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.claims = {}  # Dictionary to store claim widgets
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
        load_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        load_button.clicked.connect(self.load_pdf)
        controls_layout.addWidget(load_button)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        
        # Style for zoom buttons
        zoom_button_style = """
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """
        
        self.zoom_out_btn = QPushButton('-')
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_out_btn.setStyleSheet(zoom_button_style)
        self.zoom_out_btn.setToolTip("Zoom Out (Ctrl+-)")
        
        self.zoom_in_btn = QPushButton('+')
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_in_btn.setStyleSheet(zoom_button_style)
        self.zoom_in_btn.setToolTip("Zoom In (Ctrl++)")
        
        self.fit_width_btn = QPushButton('Fit')
        self.fit_width_btn.setFixedWidth(40)
        self.fit_width_btn.clicked.connect(self.fit_to_width)
        self.fit_width_btn.setStyleSheet(zoom_button_style)
        self.fit_width_btn.setToolTip("Fit to Width (Ctrl+0)")
        
        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.fit_width_btn)
        
        controls_layout.addLayout(zoom_layout)
        controls_layout.addStretch()
        
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
        selection_label_header.setStyleSheet("QLabel { font-weight: bold; font-size: 12px; }")
        right_layout.addWidget(selection_label_header)

        self.selection_label = QLabel("Selected text will appear here")
        self.selection_label.setWordWrap(True)
        self.selection_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #dee2e6;
                min-height: 60px;
            }
        """)
        self.selection_label.setMinimumHeight(100)
        self.selection_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        right_layout.addWidget(self.selection_label)

        # Selection control buttons
        selection_buttons_layout = QHBoxLayout()
        
        # Clear button
        self.clear_button = QPushButton('Clear Selection')
        self.clear_button.clicked.connect(self.clear_selection)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        # Preview button
        self.preview_button = QPushButton('Preview & Edit')
        self.preview_button.clicked.connect(self.preview_selection)
        self.preview_button.setEnabled(False)
        self.preview_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        selection_buttons_layout.addWidget(self.clear_button)
        selection_buttons_layout.addWidget(self.preview_button)
        right_layout.addLayout(selection_buttons_layout)

        # Action buttons
        action_buttons_layout = QHBoxLayout()
        
        self.add_claim_button = QPushButton('Add as Claim')
        self.add_claim_button.clicked.connect(self.add_claim)
        self.add_claim_button.setEnabled(False)
        self.add_claim_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
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
                background-color: #ffc107;
                color: black;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        action_buttons_layout.addWidget(self.add_claim_button)
        action_buttons_layout.addWidget(self.add_evidence_button)
        right_layout.addLayout(action_buttons_layout)

        # Claims and Evidence Section
        claims_evidence_label = QLabel('Claims and Evidence:')
        claims_evidence_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 14px;
                margin-top: 10px;
                color: #2c3e50;
            }
        """)
        right_layout.addWidget(claims_evidence_label)

        # Create a widget for claims and evidence
        claims_evidence_widget = QWidget()
        claims_evidence_layout = QVBoxLayout(claims_evidence_widget)
        claims_evidence_layout.setSpacing(10)
        
        # Scrollable area for claims and evidence
        claims_scroll = QScrollArea()
        claims_scroll.setWidget(claims_evidence_widget)
        claims_scroll.setWidgetResizable(True)
        claims_scroll.setMinimumHeight(300)
        claims_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: white;
            }
        """)
        
        self.claims_evidence_layout = claims_evidence_layout
        right_layout.addWidget(claims_scroll)

        # Claim selector for new evidence
        selector_label = QLabel('Select Claim for New Evidence:')
        selector_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                margin-top: 10px;
                color: #2c3e50;
            }
        """)
        right_layout.addWidget(selector_label)
        
        self.claim_selector = QComboBox()
        self.claim_selector.setMinimumWidth(200)
        self.claim_selector.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
            QComboBox:hover {
                border-color: #80bdff;
            }
        """)
        right_layout.addWidget(self.claim_selector)

        # Save button
        save_button = QPushButton('Save Annotations')
        save_button.clicked.connect(self.save_annotations)
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
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
                background-color: #f8f9fa;
                padding: 5px;
                border-top: 1px solid #dee2e6;
            }
        """)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                self.zoom_in()
            elif event.key() == Qt.Key_Minus:
                self.zoom_out()
            elif event.key() == Qt.Key_0:
                self.fit_to_width()
        super().keyPressEvent(event)

    def zoom_in(self):
        self.pdf_viewer.zoom_in()

    def zoom_out(self):
        self.pdf_viewer.zoom_out()

    def fit_to_width(self):
        self.pdf_viewer.fit_to_width()
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
                
                # Clear existing claims and evidence
                for claim_text in list(self.claims.keys()):
                    self.claims_evidence_layout.removeWidget(self.claims[claim_text])
                    self.claims[claim_text].deleteLater()
                self.claims.clear()
                self.claim_selector.clear()
                
                # Fit to width after loading
                QApplication.processEvents()
                self.pdf_viewer.fit_to_width()
        except Exception as e:
            self.statusBar().showMessage(f'Error loading PDF: {str(e)}')

    def handle_text_selected(self, text):
        self.selected_text = text
        display_text = text[:200] + "..." if len(text) > 200 else text
        self.selection_label.setText(f"Selected: {display_text}")
        self.preview_button.setEnabled(True)
        self.add_claim_button.setEnabled(False)
        self.add_evidence_button.setEnabled(False)

    def clear_selection(self):
        self.pdf_viewer.clear_selections()
        self.selected_text = ""
        self.selection_label.setText("Selected text will appear here")
        self.preview_button.setEnabled(False)
        self.add_claim_button.setEnabled(False)
        self.add_evidence_button.setEnabled(False)
        self.statusBar().showMessage('Selection cleared')

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
            try:
                # Create new claim widget
                claim_widget = ClaimWidget(self.selected_text)
                self.claims_evidence_layout.addWidget(claim_widget)
                
                # Add to claim selector
                self.claim_selector.addItem(self.selected_text)
                
                # Store reference to claim widget
                self.claims[self.selected_text] = claim_widget
                
                # Clear selection
                self.selected_text = ""
                self.selection_label.setText("Selected text will appear here")
                self.preview_button.setEnabled(False)
                self.add_claim_button.setEnabled(False)
                self.add_evidence_button.setEnabled(False)
                
                self.statusBar().showMessage('Claim added successfully')
            except Exception as e:
                self.statusBar().showMessage(f'Error adding claim: {str(e)}')

    def add_evidence(self):
        if not self.selected_text:
            return

        selected_claim = self.claim_selector.currentText()
        if not selected_claim or selected_claim not in self.claims:
            self.statusBar().showMessage('Please select a claim first')
            return

        try:
            # Add evidence to the corresponding claim widget
            self.claims[selected_claim].add_evidence(self.selected_text)
            
            # Clear selection
            self.selected_text = ""
            self.selection_label.setText("Selected text will appear here")
            self.preview_button.setEnabled(False)
            self.add_claim_button.setEnabled(False)
            self.add_evidence_button.setEnabled(False)
            
            self.statusBar().showMessage('Evidence added successfully')
        except Exception as e:
            self.statusBar().showMessage(f'Error adding evidence: {str(e)}')

    def save_annotations(self):
        if not self.current_pdf_path or not self.claims:
            self.statusBar().showMessage('Nothing to save')
            return

        try:
            base_name = os.path.splitext(self.current_pdf_path)[0]
            output_file = f"{base_name}_human_annotated.json"
            
            annotations = []
            for claim_text, claim_widget in self.claims.items():
                evidences = []
                for i in range(claim_widget.evidence_list.count()):
                    evidences.append(claim_widget.evidence_list.item(i).text())
                
                annotations.append({
                    "claim": claim_text,
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
        """Clean up resources when closing the application"""
        try:
            if self.doc:
                self.doc.close()
            event.accept()
        except Exception as e:
            print(f"Error closing document: {str(e)}")
            event.accept()

def main():
    try:
        app = QApplication(sys.argv)
        
        # Set application-wide stylesheet
        app.setStyle('Fusion')  # Use Fusion style for better cross-platform appearance
        
        ex = PDFAnnotationTool()
        ex.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == '__main__':
    main()
