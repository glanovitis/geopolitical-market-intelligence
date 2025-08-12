"""
CVAT Integration Module

Handles automated import of extracted outlines as pre-annotations into CVAT
for efficient training data generation.
"""

import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time


class CVATClient:
    """Client for interacting with CVAT API."""
    
    def __init__(self, cvat_url: str, token: str):
        """
        Initialize CVAT client.
        
        Args:
            cvat_url: Base URL of CVAT instance
            token: Authentication token
        """
        self.cvat_url = cvat_url.rstrip('/')
        self.token = token
        self.headers = {
            'Authorization': f'Token {token}',
            'Content-Type': 'application/json'
        }
        self.logger = logging.getLogger(__name__)
        
    def create_project(self, name: str, labels: List[Dict[str, Any]]) -> int:
        """
        Create a new CVAT project.
        
        Args:
            name: Project name
            labels: List of label definitions
            
        Returns:
            Project ID
        """
        project_data = {
            'name': name,
            'labels': labels
        }
        
        response = requests.post(
            f'{self.cvat_url}/api/projects',
            json=project_data,
            headers=self.headers
        )
        
        if response.status_code == 201:
            project_id = response.json()['id']
            self.logger.info(f"Created project '{name}' with ID: {project_id}")
            return project_id
        else:
            raise Exception(f"Failed to create project: {response.status_code} - {response.text}")
    
    def create_task(
        self, 
        name: str, 
        project_id: Optional[int] = None,
        labels: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Create a new CVAT task.
        
        Args:
            name: Task name
            project_id: Optional project ID to associate with
            labels: Label definitions (required if no project_id)
            
        Returns:
            Task ID
        """
        task_data = {
            'name': name,
            'segment_size': 1,
        }
        
        if project_id:
            task_data['project_id'] = project_id
        elif labels:
            task_data['labels'] = labels
        else:
            # Default floorplan labels
            task_data['labels'] = [
                {'name': 'indoor', 'color': '#00ff00', 'attributes': []},
                {'name': 'outdoor', 'color': '#0000ff', 'attributes': []}
            ]
        
        response = requests.post(
            f'{self.cvat_url}/api/tasks',
            json=task_data,
            headers=self.headers
        )
        
        if response.status_code == 201:
            task_id = response.json()['id']
            self.logger.info(f"Created task '{name}' with ID: {task_id}")
            return task_id
        else:
            raise Exception(f"Failed to create task: {response.status_code} - {response.text}")
    
    def upload_data(self, task_id: int, image_path: str) -> None:
        """
        Upload image data to a CVAT task.
        
        Args:
            task_id: Task ID
            image_path: Path to image file
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Prepare file upload
        files = {
            'client_files[0]': (image_path.name, open(image_path, 'rb'), 'image/png')
        }
        
        upload_headers = {
            'Authorization': f'Token {self.token}'
        }
        
        response = requests.post(
            f'{self.cvat_url}/api/tasks/{task_id}/data',
            files=files,
            headers=upload_headers
        )
        
        if response.status_code == 202:
            self.logger.info(f"Successfully uploaded image to task {task_id}")
            # Wait for upload to complete
            self._wait_for_task_completion(task_id)
        else:
            raise Exception(f"Failed to upload data: {response.status_code} - {response.text}")
    
    def upload_annotations(self, task_id: int, annotations_path: str) -> None:
        """
        Upload annotations to a CVAT task.
        
        Args:
            task_id: Task ID
            annotations_path: Path to annotations JSON file
        """
        annotations_path = Path(annotations_path)
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        
        # Read annotations
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)
        
        # Upload annotations
        response = requests.put(
            f'{self.cvat_url}/api/tasks/{task_id}/annotations',
            json=annotations_data,
            headers=self.headers
        )
        
        if response.status_code == 202:
            self.logger.info(f"Successfully uploaded annotations to task {task_id}")
        else:
            raise Exception(f"Failed to upload annotations: {response.status_code} - {response.text}")
    
    def get_task_info(self, task_id: int) -> Dict[str, Any]:
        """
        Get information about a CVAT task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task information dictionary
        """
        response = requests.get(
            f'{self.cvat_url}/api/tasks/{task_id}',
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get task info: {response.status_code} - {response.text}")
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all projects in CVAT instance.
        
        Returns:
            List of project dictionaries
        """
        response = requests.get(
            f'{self.cvat_url}/api/projects',
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()['results']
        else:
            raise Exception(f"Failed to list projects: {response.status_code} - {response.text}")
    
    def list_tasks(self, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List tasks, optionally filtered by project.
        
        Args:
            project_id: Optional project ID to filter by
            
        Returns:
            List of task dictionaries
        """
        url = f'{self.cvat_url}/api/tasks'
        if project_id:
            url += f'?project_id={project_id}'
        
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()['results']
        else:
            raise Exception(f"Failed to list tasks: {response.status_code} - {response.text}")
    
    def _wait_for_task_completion(self, task_id: int, timeout: int = 300) -> None:
        """
        Wait for a task to complete data processing.
        
        Args:
            task_id: Task ID
            timeout: Maximum wait time in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task_info = self.get_task_info(task_id)
            status = task_info.get('status', '')
            
            if status == 'completed':
                self.logger.info(f"Task {task_id} completed successfully")
                return
            elif status == 'failed':
                raise Exception(f"Task {task_id} failed")
            
            time.sleep(5)  # Wait 5 seconds before checking again
        
        raise Exception(f"Task {task_id} did not complete within {timeout} seconds")


class FloorplanCVATImporter:
    """High-level interface for importing floorplan annotations to CVAT."""
    
    def __init__(self, cvat_client: CVATClient):
        """
        Initialize the importer.
        
        Args:
            cvat_client: Configured CVAT client
        """
        self.cvat_client = cvat_client
        self.logger = logging.getLogger(__name__)
    
    def import_floorplan(
        self,
        image_path: str,
        annotations_path: str,
        project_name: str = "Floorplan Segmentation",
        task_name: str = None
    ) -> Dict[str, int]:
        """
        Import floorplan image and annotations to CVAT.
        
        Args:
            image_path: Path to floorplan image
            annotations_path: Path to annotations JSON
            project_name: Name for CVAT project
            task_name: Name for CVAT task (auto-generated if None)
            
        Returns:
            Dictionary with project_id and task_id
        """
        image_path = Path(image_path)
        
        if task_name is None:
            task_name = f"Floorplan_{image_path.stem}"
        
        # Check if project exists or create new one
        projects = self.cvat_client.list_projects()
        project_id = None
        
        for project in projects:
            if project['name'] == project_name:
                project_id = project['id']
                self.logger.info(f"Using existing project: {project_name} (ID: {project_id})")
                break
        
        if project_id is None:
            # Create new project
            labels = [
                {'name': 'indoor', 'color': '#00ff00', 'attributes': []},
                {'name': 'outdoor', 'color': '#0000ff', 'attributes': []},
                {'name': 'architectural_elements', 'color': '#ff0000', 'attributes': []}
            ]
            project_id = self.cvat_client.create_project(project_name, labels)
        
        # Create task
        task_id = self.cvat_client.create_task(task_name, project_id)
        
        # Upload image
        self.cvat_client.upload_data(task_id, image_path)
        
        # Upload annotations
        self.cvat_client.upload_annotations(task_id, annotations_path)
        
        self.logger.info(f"Successfully imported floorplan to CVAT - Project ID: {project_id}, Task ID: {task_id}")
        
        return {
            'project_id': project_id,
            'task_id': task_id,
            'cvat_url': f"{self.cvat_client.cvat_url}/tasks/{task_id}"
        }
    
    def batch_import(
        self,
        image_annotations_pairs: List[Tuple[str, str]],
        project_name: str = "Floorplan Batch Import"
    ) -> List[Dict[str, Any]]:
        """
        Batch import multiple floorplan images and annotations.
        
        Args:
            image_annotations_pairs: List of (image_path, annotations_path) tuples
            project_name: Name for CVAT project
            
        Returns:
            List of import results
        """
        results = []
        
        # Create or get project
        projects = self.cvat_client.list_projects()
        project_id = None
        
        for project in projects:
            if project['name'] == project_name:
                project_id = project['id']
                break
        
        if project_id is None:
            labels = [
                {'name': 'indoor', 'color': '#00ff00', 'attributes': []},
                {'name': 'outdoor', 'color': '#0000ff', 'attributes': []},
                {'name': 'architectural_elements', 'color': '#ff0000', 'attributes': []}
            ]
            project_id = self.cvat_client.create_project(project_name, labels)
        
        for i, (image_path, annotations_path) in enumerate(image_annotations_pairs):
            try:
                image_name = Path(image_path).stem
                task_name = f"{project_name}_Task_{i+1}_{image_name}"
                
                result = self.import_floorplan(
                    image_path,
                    annotations_path,
                    project_name,
                    task_name
                )
                result['status'] = 'success'
                result['image_path'] = image_path
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to import {image_path}: {str(e)}")
                results.append({
                    'status': 'failed',
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results