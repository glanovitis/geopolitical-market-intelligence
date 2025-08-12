"""
Example usage script for the PDF outline extraction pipeline.

This script demonstrates how to use the pipeline components to process
floorplan PDFs and import them to CVAT.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pdf_processing.outline_extractor import PDFOutlineExtractor
from src.cvat_integration.cvat_client import CVATClient, FloorplanCVATImporter
from server.scripts.pdf_processor import FloorplanProcessor


def example_basic_processing():
    """Example: Basic PDF processing without CVAT integration."""
    print("=" * 60)
    print("EXAMPLE 1: Basic PDF Processing")
    print("=" * 60)
    
    # Initialize extractor
    extractor = PDFOutlineExtractor(dpi=300)
    
    # For this example, we'll create mock data since we don't have a real PDF
    print("Creating mock outline data...")
    
    mock_outline_data = {
        'image_width': 1200,
        'image_height': 800,
        'outlines': [
            {
                'id': 0,
                'type': 'polygon',
                'points': [100, 100, 400, 100, 400, 300, 100, 300],
                'category': 'indoor',
                'confidence': 1.0,
                'area': 60000
            },
            {
                'id': 1,
                'type': 'polygon',
                'points': [50, 50, 500, 50, 500, 400, 50, 400],
                'category': 'outdoor',
                'confidence': 0.95,
                'area': 157500
            }
        ],
        'vector_paths': [],
        'source_pdf': 'example_floorplan.pdf'
    }
    
    # Convert to CVAT format
    print("Converting to CVAT format...")
    cvat_data = extractor.convert_to_cvat_format(mock_outline_data, "Example_Floorplan")
    
    print(f"✓ Processed {len(mock_outline_data['outlines'])} outlines")
    print(f"✓ Task name: {cvat_data['meta']['task']['name']}")
    print(f"✓ Labels available: {[label['name'] for label in cvat_data['meta']['task']['labels']]}")
    
    # Save annotations to file
    output_dir = project_root / "examples" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_path = extractor.save_annotations(cvat_data, str(output_dir))
    print(f"✓ Annotations saved to: {annotations_path}")
    
    return cvat_data


def example_cvat_integration():
    """Example: CVAT integration (requires CVAT instance)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CVAT Integration")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    cvat_url = os.getenv('CVAT_URL', 'http://localhost:8080')
    cvat_token = os.getenv('CVAT_TOKEN')
    
    if not cvat_token or cvat_token == 'your_cvat_token_here':
        print("⚠️  CVAT token not configured. Skipping CVAT integration example.")
        print("   Set CVAT_TOKEN in your .env file to test CVAT integration.")
        return
    
    try:
        # Initialize CVAT client
        cvat_client = CVATClient(cvat_url, cvat_token)
        importer = FloorplanCVATImporter(cvat_client)
        
        # Test connection
        projects = cvat_client.list_projects()
        print(f"✓ Connected to CVAT. Found {len(projects)} existing projects.")
        
        # Example project creation (commented out to avoid creating real projects)
        # project_name = "Example Floorplan Project"
        # labels = [
        #     {'name': 'indoor', 'color': '#00ff00', 'attributes': []},
        #     {'name': 'outdoor', 'color': '#0000ff', 'attributes': []}
        # ]
        # project_id = cvat_client.create_project(project_name, labels)
        # print(f"✓ Created project: {project_name} (ID: {project_id})")
        
        print("✓ CVAT integration is working")
        
    except Exception as e:
        print(f"❌ CVAT integration failed: {str(e)}")
        print("   Check your CVAT_URL and CVAT_TOKEN configuration.")


def example_server_pipeline():
    """Example: Server pipeline usage."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Server Pipeline")
    print("=" * 60)
    
    load_dotenv()
    
    cvat_url = os.getenv('CVAT_URL', 'http://localhost:8080')
    cvat_token = os.getenv('CVAT_TOKEN', 'demo_token')
    
    # Initialize processor
    processor = FloorplanProcessor(
        cvat_url=cvat_url,
        cvat_token=cvat_token,
        uploads_dir="examples/uploads"
    )
    
    print("✓ FloorplanProcessor initialized")
    print(f"  - CVAT URL: {cvat_url}")
    print(f"  - Uploads directory: {processor.uploads_dir}")
    
    # Example of how to use the processor
    print("\nProcessor capabilities:")
    print("  - process_pdf(pdf_path, project_name, customer_info)")
    print("  - batch_process(pdf_directory, project_name)")
    print("  - get_processing_status(process_id)")
    print("  - list_processing_jobs()")
    print("  - cleanup_old_jobs(days_old)")
    
    # List any existing jobs
    jobs = processor.list_processing_jobs()
    print(f"\nFound {len(jobs)} existing processing jobs")


def example_api_usage():
    """Example: API usage with curl commands."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: API Usage")
    print("=" * 60)
    
    print("To use the REST API, start the server:")
    print("  cd server")
    print("  python api_server.py")
    print("\nThen use these curl commands:")
    print()
    print("1. Health check:")
    print("   curl http://localhost:8000/api/health")
    print()
    print("2. Process a PDF:")
    print("   curl -X POST http://localhost:8000/api/process-floorplan \\")
    print("     -F 'pdf_file=@your_floorplan.pdf' \\")
    print("     -F 'project_name=My Project' \\")
    print("     -F 'customer_name=John Doe'")
    print()
    print("3. Check processing status:")
    print("   curl http://localhost:8000/api/status/{process_id}")
    print()
    print("4. List all jobs:")
    print("   curl http://localhost:8000/api/jobs")
    print()
    print("5. Batch process (multiple files):")
    print("   curl -X POST http://localhost:8000/api/batch-process \\")
    print("     -F 'pdf_files=@file1.pdf' \\")
    print("     -F 'pdf_files=@file2.pdf' \\")
    print("     -F 'project_name=Batch Project'")


def main():
    """Run all examples."""
    print("PDF Outline Extraction to CVAT Import Pipeline - Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_processing()
    example_cvat_integration()
    example_server_pipeline()
    example_api_usage()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Configure your .env file with CVAT credentials")
    print("2. Start the API server: cd server && python api_server.py")
    print("3. Try processing a real PDF floorplan")
    print("4. Check the generated annotations in CVAT")


if __name__ == "__main__":
    main()