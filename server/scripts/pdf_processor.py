"""
Server-side processing pipeline for PDF floorplan processing.

This script handles automated file processing, storage organization,
and CVAT project/task creation.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime

from src.pdf_processing.outline_extractor import PDFOutlineExtractor
from src.cvat_integration.cvat_client import CVATClient, FloorplanCVATImporter


class FloorplanProcessor:
    """Main processor for PDF floorplan extraction and CVAT import pipeline."""
    
    def __init__(
        self,
        cvat_url: str,
        cvat_token: str,
        uploads_dir: str = "server/uploads",
        dpi: int = 300
    ):
        """
        Initialize the floorplan processor.
        
        Args:
            cvat_url: CVAT instance URL
            cvat_token: CVAT authentication token
            uploads_dir: Directory for file uploads and processing
            dpi: DPI for PDF to image conversion
        """
        self.cvat_url = cvat_url
        self.cvat_token = cvat_token
        self.uploads_dir = Path(uploads_dir)
        self.dpi = dpi
        
        # Create upload directories
        self.pdfs_dir = self.uploads_dir / "pdfs"
        self.images_dir = self.uploads_dir / "images"
        self.annotations_dir = self.uploads_dir / "annotations"
        
        for dir_path in [self.pdfs_dir, self.images_dir, self.annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = PDFOutlineExtractor(dpi=dpi)
        self.cvat_client = CVATClient(cvat_url, cvat_token)
        self.importer = FloorplanCVATImporter(self.cvat_client)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def process_pdf(
        self,
        pdf_path: str,
        project_name: str = "Floorplan Processing",
        customer_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single PDF floorplan.
        
        Args:
            pdf_path: Path to PDF file
            project_name: Name for CVAT project
            customer_info: Optional customer information
            
        Returns:
            Processing result dictionary
        """
        pdf_path = Path(pdf_path)
        process_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Starting processing for PDF: {pdf_path.name} (ID: {process_id})")
        
        try:
            # Create processing directory
            process_dir = self.uploads_dir / f"process_{process_id}"
            process_dir.mkdir(exist_ok=True)
            
            # Copy PDF to processing directory
            pdf_copy = process_dir / pdf_path.name
            shutil.copy2(pdf_path, pdf_copy)
            
            # Extract outline data
            self.logger.info("Extracting outlines from PDF...")
            outline_data = self.extractor.extract_outlines_from_pdf(pdf_copy)
            
            # Convert to CVAT format
            task_name = f"Floorplan_{pdf_path.stem}_{process_id[:8]}"
            cvat_data = self.extractor.convert_to_cvat_format(outline_data, task_name)
            
            # Save image
            self.logger.info("Converting PDF to image...")
            image_path = self.extractor.save_image(pdf_copy, self.images_dir)
            
            # Save annotations
            self.logger.info("Saving annotations...")
            annotations_filename = f"{pdf_path.stem}_{process_id[:8]}_annotations.json"
            annotations_path = self.extractor.save_annotations(
                cvat_data, 
                self.annotations_dir, 
                annotations_filename
            )
            
            # Import to CVAT
            self.logger.info("Importing to CVAT...")
            cvat_result = self.importer.import_floorplan(
                image_path,
                annotations_path,
                project_name,
                task_name
            )
            
            # Create processing metadata
            metadata = {
                'process_id': process_id,
                'timestamp': timestamp,
                'pdf_path': str(pdf_path),
                'image_path': image_path,
                'annotations_path': annotations_path,
                'customer_info': customer_info or {},
                'outline_count': len(outline_data['outlines']),
                'cvat_result': cvat_result,
                'status': 'completed'
            }
            
            # Save metadata
            metadata_path = process_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Processing completed successfully for {pdf_path.name}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Processing failed for {pdf_path.name}: {str(e)}")
            error_metadata = {
                'process_id': process_id,
                'timestamp': timestamp,
                'pdf_path': str(pdf_path),
                'error': str(e),
                'status': 'failed'
            }
            return error_metadata
    
    def batch_process(
        self,
        pdf_directory: str,
        project_name: str = "Batch Floorplan Processing"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDFs in batch.
        
        Args:
            pdf_directory: Directory containing PDF files
            project_name: Name for CVAT project
            
        Returns:
            List of processing results
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files for batch processing")
        
        results = []
        for pdf_file in pdf_files:
            result = self.process_pdf(pdf_file, project_name)
            results.append(result)
        
        # Create batch summary
        successful = [r for r in results if r.get('status') == 'completed']
        failed = [r for r in results if r.get('status') == 'failed']
        
        batch_summary = {
            'total_files': len(pdf_files),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(pdf_files) * 100 if pdf_files else 0,
            'results': results
        }
        
        # Save batch summary
        batch_dir = self.uploads_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_dir.mkdir(exist_ok=True)
        
        summary_path = batch_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        self.logger.info(f"Batch processing completed: {len(successful)}/{len(pdf_files)} successful")
        return results
    
    def get_processing_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a processing job.
        
        Args:
            process_id: Processing job ID
            
        Returns:
            Processing status dictionary or None if not found
        """
        process_dir = self.uploads_dir / f"process_{process_id}"
        metadata_path = process_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def list_processing_jobs(self) -> List[Dict[str, Any]]:
        """
        List all processing jobs.
        
        Returns:
            List of processing job metadata
        """
        jobs = []
        
        for process_dir in self.uploads_dir.glob("process_*"):
            metadata_path = process_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    jobs.append(json.load(f))
        
        # Sort by timestamp (newest first)
        jobs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return jobs
    
    def cleanup_old_jobs(self, days_old: int = 30) -> int:
        """
        Clean up processing jobs older than specified days.
        
        Args:
            days_old: Number of days after which to delete jobs
            
        Returns:
            Number of jobs cleaned up
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0
        
        for process_dir in self.uploads_dir.glob("process_*"):
            metadata_path = process_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                job_date = datetime.fromisoformat(metadata.get('timestamp', ''))
                if job_date < cutoff_date:
                    shutil.rmtree(process_dir)
                    cleaned_count += 1
                    self.logger.info(f"Cleaned up old job: {metadata.get('process_id')}")
        
        return cleaned_count


def main():
    """Example usage of the FloorplanProcessor."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    cvat_url = os.getenv('CVAT_URL', 'http://localhost:8080')
    cvat_token = os.getenv('CVAT_TOKEN', 'your_cvat_token_here')
    
    # Initialize processor
    processor = FloorplanProcessor(cvat_url, cvat_token)
    
    # Example: Process a single PDF
    # pdf_path = "path/to/your/floorplan.pdf"
    # result = processor.process_pdf(pdf_path)
    # print(f"Processing result: {result}")
    
    # Example: Batch process PDFs
    # pdf_directory = "path/to/pdf/directory"
    # results = processor.batch_process(pdf_directory)
    # print(f"Batch processing completed: {len(results)} files processed")


if __name__ == "__main__":
    main()