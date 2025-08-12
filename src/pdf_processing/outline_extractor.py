"""
PDF Outline Extraction Module

Extracts vector paths from PDF floorplan outlines and converts them to JSON format
compatible with segmentation annotations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import fitz  # PyMuPDF
import pdf2image
import cv2
import numpy as np
from PIL import Image


class PDFOutlineExtractor:
    """Extract outline data from PDF floorplans."""
    
    def __init__(self, dpi: int = 300):
        """
        Initialize the PDF outline extractor.
        
        Args:
            dpi: Resolution for PDF to image conversion
        """
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
    def extract_outlines_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract outline data from PDF floorplan.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted outline data
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to high-resolution image
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=self.dpi,
            fmt='PNG'
        )
        
        if not images:
            raise ValueError("No pages found in PDF")
            
        # Process the first page (assuming single-page floorplan)
        image = images[0]
        image_array = np.array(image)
        
        # Extract outlines from image
        outlines = self._extract_outlines_from_image(image_array)
        
        # Extract vector paths from PDF
        vector_paths = self._extract_vector_paths_from_pdf(pdf_path)
        
        return {
            'image_width': image.width,
            'image_height': image.height,
            'outlines': outlines,
            'vector_paths': vector_paths,
            'source_pdf': str(pdf_path)
        }
    
    def _extract_outlines_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract outlines from image using computer vision techniques.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of outline dictionaries
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        outlines = []
        for i, contour in enumerate(contours):
            # Filter out small contours
            if cv2.contourArea(contour) < 1000:
                continue
                
            # Approximate contour to reduce points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to list of points
            points = []
            for point in approx:
                points.extend([int(point[0][0]), int(point[0][1])])
            
            outline = {
                'id': i,
                'type': 'polygon',
                'points': points,
                'category': self._detect_category(contour, image.shape),
                'confidence': 1.0,
                'area': cv2.contourArea(contour)
            }
            outlines.append(outline)
        
        return outlines
    
    def _extract_vector_paths_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract vector paths directly from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of vector path dictionaries
        """
        doc = fitz.open(pdf_path)
        vector_paths = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get drawing commands
                drawings = page.get_drawings()
                
                for i, drawing in enumerate(drawings):
                    if 'items' in drawing:
                        path_points = []
                        for item in drawing['items']:
                            if item[0] == 'l':  # line
                                path_points.extend([item[1].x, item[1].y, item[2].x, item[2].y])
                            elif item[0] == 'c':  # curve
                                # Approximate curve with line segments
                                path_points.extend([item[1].x, item[1].y, item[4].x, item[4].y])
                        
                        if path_points:
                            vector_paths.append({
                                'id': f"vector_{page_num}_{i}",
                                'type': 'vector_path',
                                'points': path_points,
                                'stroke_width': drawing.get('width', 1.0),
                                'color': drawing.get('color', '#000000')
                            })
        
        finally:
            doc.close()
        
        return vector_paths
    
    def _detect_category(self, contour: np.ndarray, image_shape: Tuple[int, ...]) -> str:
        """
        Detect if contour represents indoor or outdoor space.
        
        Args:
            contour: OpenCV contour
            image_shape: Shape of the image
            
        Returns:
            Category string ('indoor' or 'outdoor')
        """
        # Simple heuristic: larger contours near image borders are likely outdoor
        x, y, w, h = cv2.boundingRect(contour)
        image_height, image_width = image_shape[:2]
        
        # Check if contour touches image borders
        border_threshold = 50
        touches_border = (
            x < border_threshold or 
            y < border_threshold or 
            x + w > image_width - border_threshold or 
            y + h > image_height - border_threshold
        )
        
        area = cv2.contourArea(contour)
        image_area = image_width * image_height
        area_ratio = area / image_area
        
        # Large contours touching borders are likely outdoor
        if touches_border and area_ratio > 0.1:
            return 'outdoor'
        else:
            return 'indoor'
    
    def convert_to_cvat_format(
        self, 
        outline_data: Dict[str, Any], 
        task_name: str = "floorplan_task"
    ) -> Dict[str, Any]:
        """
        Convert outline data to CVAT-compatible annotation format.
        
        Args:
            outline_data: Extracted outline data
            task_name: Name for the CVAT task
            
        Returns:
            CVAT-compatible annotation dictionary
        """
        annotations = []
        
        for outline in outline_data['outlines']:
            # Normalize points to image dimensions
            normalized_points = self._normalize_points(
                outline['points'],
                outline_data['image_width'],
                outline_data['image_height']
            )
            
            annotation = {
                'id': outline['id'],
                'type': 'polygon',
                'occluded': False,
                'points': normalized_points,
                'label': outline['category'],
                'attributes': [
                    {'name': 'confidence', 'value': str(outline['confidence'])},
                    {'name': 'area', 'value': str(outline['area'])}
                ]
            }
            annotations.append(annotation)
        
        cvat_format = {
            'version': '1.1',
            'meta': {
                'task': {
                    'name': task_name,
                    'size': 1,
                    'mode': 'annotation',
                    'labels': [
                        {'name': 'indoor', 'color': '#00ff00', 'attributes': []},
                        {'name': 'outdoor', 'color': '#0000ff', 'attributes': []}
                    ]
                }
            },
            'annotations': annotations,
            'tracks': []
        }
        
        return cvat_format
    
    def _normalize_points(
        self, 
        points: List[float], 
        image_width: int, 
        image_height: int
    ) -> List[float]:
        """
        Normalize points to relative coordinates [0-1].
        
        Args:
            points: List of x,y coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Normalized points list
        """
        normalized = []
        for i in range(0, len(points), 2):
            x = points[i] / image_width
            y = points[i + 1] / image_height
            normalized.extend([x, y])
        
        return normalized
    
    def save_image(self, pdf_path: str, output_dir: str) -> str:
        """
        Convert PDF to high-resolution image and save.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save the image
            
        Returns:
            Path to saved image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=self.dpi,
            fmt='PNG'
        )
        
        if not images:
            raise ValueError("No pages found in PDF")
        
        pdf_name = Path(pdf_path).stem
        image_path = output_dir / f"{pdf_name}_floorplan.png"
        
        # Save the first page
        images[0].save(image_path, 'PNG')
        
        self.logger.info(f"Saved image to: {image_path}")
        return str(image_path)
    
    def save_annotations(
        self, 
        cvat_data: Dict[str, Any], 
        output_dir: str, 
        filename: str = None
    ) -> str:
        """
        Save CVAT annotations to JSON file.
        
        Args:
            cvat_data: CVAT-formatted annotation data
            output_dir: Directory to save annotations
            filename: Optional filename (will generate if not provided)
            
        Returns:
            Path to saved annotation file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{cvat_data['meta']['task']['name']}_annotations.json"
        
        annotation_path = output_dir / filename
        
        with open(annotation_path, 'w') as f:
            json.dump(cvat_data, f, indent=2)
        
        self.logger.info(f"Saved annotations to: {annotation_path}")
        return str(annotation_path)