"""
FastAPI server for PDF floorplan processing pipeline.

Provides REST API endpoints for frontend integration and batch processing.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our processing modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from server.scripts.pdf_processor import FloorplanProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Floorplan Processing API",
    description="API for extracting floorplan outlines from PDFs and importing to CVAT",
    version="1.0.0"
)

# Configuration
CVAT_URL = os.getenv('CVAT_URL', 'http://localhost:8080')
CVAT_TOKEN = os.getenv('CVAT_TOKEN', 'your_cvat_token_here')
UPLOADS_DIR = os.getenv('UPLOADS_DIR', 'server/uploads')

# Initialize processor
processor = FloorplanProcessor(CVAT_URL, CVAT_TOKEN, UPLOADS_DIR)


# Pydantic models
class ProcessingRequest(BaseModel):
    project_name: str = "Floorplan Processing"
    customer_info: Optional[Dict[str, Any]] = None


class BatchProcessingRequest(BaseModel):
    project_name: str = "Batch Floorplan Processing"


class ProcessingResponse(BaseModel):
    process_id: str
    status: str
    message: str
    cvat_task_id: Optional[int] = None
    cvat_url: Optional[str] = None


class StatusResponse(BaseModel):
    process_id: str
    status: str
    timestamp: str
    pdf_path: Optional[str] = None
    image_path: Optional[str] = None
    annotations_path: Optional[str] = None
    cvat_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PDF Floorplan Processing API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/api/process-floorplan",
            "batch": "/api/batch-process",
            "status": "/api/status/{process_id}",
            "jobs": "/api/jobs"
        }
    }


@app.post("/api/process-floorplan", response_model=ProcessingResponse)
async def process_floorplan(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    project_name: str = "Floorplan Processing",
    customer_name: str = "",
    customer_email: str = ""
):
    """
    Process a single PDF floorplan.
    
    Args:
        pdf_file: Uploaded PDF file
        project_name: Name for CVAT project
        customer_name: Customer name
        customer_email: Customer email
        
    Returns:
        Processing response with job ID and status
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await pdf_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Prepare customer info
        customer_info = {
            'name': customer_name,
            'email': customer_email,
            'filename': pdf_file.filename
        }
        
        # Start background processing
        result = processor.process_pdf(
            temp_path,
            project_name,
            customer_info
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if result.get('status') == 'completed':
            cvat_result = result.get('cvat_result', {})
            return ProcessingResponse(
                process_id=result['process_id'],
                status="completed",
                message="PDF processed successfully",
                cvat_task_id=cvat_result.get('task_id'),
                cvat_url=cvat_result.get('cvat_url')
            )
        else:
            return ProcessingResponse(
                process_id=result['process_id'],
                status="failed",
                message=f"Processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/batch-process")
async def batch_process(
    background_tasks: BackgroundTasks,
    pdf_files: List[UploadFile] = File(...),
    project_name: str = "Batch Floorplan Processing"
):
    """
    Process multiple PDF files in batch.
    
    Args:
        pdf_files: List of uploaded PDF files
        project_name: Name for CVAT project
        
    Returns:
        Batch processing results
    """
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate all files are PDFs
    for pdf_file in pdf_files:
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"File {pdf_file.filename} must be a PDF"
            )
    
    try:
        # Create temporary directory for batch processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Save all files
            for pdf_file in pdf_files:
                content = await pdf_file.read()
                temp_file_path = temp_dir_path / pdf_file.filename
                with open(temp_file_path, 'wb') as f:
                    f.write(content)
            
            # Process batch
            results = processor.batch_process(temp_dir, project_name)
            
            return {
                "message": "Batch processing completed",
                "total_files": len(pdf_files),
                "successful": len([r for r in results if r.get('status') == 'completed']),
                "failed": len([r for r in results if r.get('status') == 'failed']),
                "results": results
            }
            
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/api/status/{process_id}", response_model=StatusResponse)
async def get_processing_status(process_id: str):
    """
    Get status of a processing job.
    
    Args:
        process_id: Processing job ID
        
    Returns:
        Processing status information
    """
    status = processor.get_processing_status(process_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    return StatusResponse(**status)


@app.get("/api/jobs")
async def list_processing_jobs():
    """
    List all processing jobs.
    
    Returns:
        List of processing job information
    """
    jobs = processor.list_processing_jobs()
    return {
        "total_jobs": len(jobs),
        "jobs": jobs
    }


@app.delete("/api/cleanup")
async def cleanup_old_jobs(days_old: int = 30):
    """
    Clean up processing jobs older than specified days.
    
    Args:
        days_old: Number of days after which to delete jobs
        
    Returns:
        Cleanup result
    """
    try:
        cleaned_count = processor.cleanup_old_jobs(days_old)
        return {
            "message": f"Cleanup completed",
            "jobs_cleaned": cleaned_count
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test CVAT connection
        projects = processor.cvat_client.list_projects()
        cvat_status = "connected"
    except Exception as e:
        cvat_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "cvat_status": cvat_status,
        "uploads_dir": str(processor.uploads_dir),
        "processor_ready": True
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )