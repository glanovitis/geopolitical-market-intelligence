"""
Tests for PDF processing pipeline.

Simple tests to verify the basic functionality of the PDF outline extraction
and CVAT integration modules.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

from src.pdf_processing.outline_extractor import PDFOutlineExtractor
from src.cvat_integration.cvat_client import CVATClient


class TestPDFOutlineExtractor:
    """Test PDF outline extraction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PDFOutlineExtractor(dpi=150)  # Lower DPI for testing
    
    def test_initialization(self):
        """Test extractor initialization."""
        assert self.extractor.dpi == 150
        assert self.extractor.logger is not None
    
    def test_normalize_points(self):
        """Test point normalization."""
        points = [100, 200, 300, 400]  # x1, y1, x2, y2
        normalized = self.extractor._normalize_points(points, 400, 800)
        
        expected = [0.25, 0.25, 0.75, 0.5]  # Normalized coordinates
        assert normalized == expected
    
    def test_detect_category_indoor(self):
        """Test category detection for indoor spaces."""
        # Create a mock contour (small, not touching borders)
        contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]])
        image_shape = (600, 800)  # height, width
        
        category = self.extractor._detect_category(contour, image_shape)
        assert category == 'indoor'
    
    def test_detect_category_outdoor(self):
        """Test category detection for outdoor spaces."""
        # Create a mock contour (large, touching borders)
        contour = np.array([[[10, 10]], [[790, 10]], [[790, 590]], [[10, 590]]])
        image_shape = (600, 800)  # height, width
        
        category = self.extractor._detect_category(contour, image_shape)
        assert category == 'outdoor'
    
    def test_convert_to_cvat_format(self):
        """Test conversion to CVAT format."""
        outline_data = {
            'image_width': 800,
            'image_height': 600,
            'outlines': [
                {
                    'id': 0,
                    'type': 'polygon',
                    'points': [100, 100, 200, 100, 200, 200, 100, 200],
                    'category': 'indoor',
                    'confidence': 1.0,
                    'area': 10000
                }
            ],
            'vector_paths': [],
            'source_pdf': 'test.pdf'
        }
        
        cvat_data = self.extractor.convert_to_cvat_format(outline_data, "test_task")
        
        assert cvat_data['version'] == '1.1'
        assert cvat_data['meta']['task']['name'] == 'test_task'
        assert len(cvat_data['annotations']) == 1
        assert cvat_data['annotations'][0]['label'] == 'indoor'
        assert cvat_data['annotations'][0]['type'] == 'polygon'
    
    @patch('pdf2image.convert_from_path')
    def test_save_image(self, mock_convert):
        """Test image saving functionality."""
        # Mock PDF conversion
        mock_image = Mock()
        mock_convert.return_value = [mock_image]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = self.extractor.save_image('test.pdf', temp_dir)
            
            expected_path = str(Path(temp_dir) / 'test_floorplan.png')
            assert result_path == expected_path
            mock_image.save.assert_called_once()
    
    def test_save_annotations(self):
        """Test annotation saving functionality."""
        cvat_data = {
            'version': '1.1',
            'meta': {'task': {'name': 'test_task'}},
            'annotations': []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = self.extractor.save_annotations(cvat_data, temp_dir)
            
            expected_path = str(Path(temp_dir) / 'test_task_annotations.json')
            assert result_path == expected_path
            
            # Verify file content
            with open(result_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data == cvat_data


class TestCVATClient:
    """Test CVAT client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = CVATClient('http://test-cvat.com', 'test_token')
    
    def test_initialization(self):
        """Test client initialization."""
        assert self.client.cvat_url == 'http://test-cvat.com'
        assert self.client.token == 'test_token'
        assert 'Authorization' in self.client.headers
        assert self.client.headers['Authorization'] == 'Token test_token'
    
    @patch('requests.post')
    def test_create_project_success(self, mock_post):
        """Test successful project creation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 123}
        mock_post.return_value = mock_response
        
        labels = [{'name': 'indoor', 'color': '#00ff00'}]
        project_id = self.client.create_project('test_project', labels)
        
        assert project_id == 123
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_create_project_failure(self, mock_post):
        """Test project creation failure."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = 'Bad request'
        mock_post.return_value = mock_response
        
        labels = [{'name': 'indoor', 'color': '#00ff00'}]
        
        with pytest.raises(Exception) as exc_info:
            self.client.create_project('test_project', labels)
        
        assert 'Failed to create project' in str(exc_info.value)
    
    @patch('requests.post')
    def test_create_task_success(self, mock_post):
        """Test successful task creation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 456}
        mock_post.return_value = mock_response
        
        task_id = self.client.create_task('test_task', project_id=123)
        
        assert task_id == 456
        mock_post.assert_called_once()
    
    @patch('requests.get')
    def test_get_task_info_success(self, mock_get):
        """Test successful task info retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'id': 456, 'name': 'test_task'}
        mock_get.return_value = mock_response
        
        task_info = self.client.get_task_info(456)
        
        assert task_info['id'] == 456
        assert task_info['name'] == 'test_task'
    
    @patch('requests.get')
    def test_list_projects(self, mock_get):
        """Test listing projects."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{'id': 1, 'name': 'project1'}, {'id': 2, 'name': 'project2'}]
        }
        mock_get.return_value = mock_response
        
        projects = self.client.list_projects()
        
        assert len(projects) == 2
        assert projects[0]['name'] == 'project1'


def test_import_functionality():
    """Test that all modules can be imported successfully."""
    from src.pdf_processing.outline_extractor import PDFOutlineExtractor
    from src.cvat_integration.cvat_client import CVATClient, FloorplanCVATImporter
    
    # Basic instantiation test
    extractor = PDFOutlineExtractor()
    client = CVATClient('http://test.com', 'token')
    importer = FloorplanCVATImporter(client)
    
    assert extractor is not None
    assert client is not None
    assert importer is not None


if __name__ == '__main__':
    pytest.main([__file__])