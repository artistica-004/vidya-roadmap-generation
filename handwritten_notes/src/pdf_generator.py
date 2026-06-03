import os
import logging
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class PDFRenderError(Exception):
    """Custom exception for PDF generation failures."""
    pass

def render_pdf(roadmap, output_path: Path):
    """
    Renders a CourseRoadmap object into a PDF using an HTML template.
    
    Args:
        roadmap: The Pydantic model (CourseRoadmap) containing the data.
        output_path: Path object where the final PDF will be saved.
    """
    try:
        # 1. Determine the directory of this file (src folder)
        # This ensures we find notebook.html regardless of where we run from
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Setup Jinja2 Environment
        # We tell it to look for templates specifically in this folder
        env = Environment(loader=FileSystemLoader(current_dir))
        
        try:
            template = env.get_template('notebook.html')
        except Exception as e:
            logger.error(f"Could not find notebook.html in {current_dir}")
            raise PDFRenderError(f"Template file missing: {str(e)}")

        # 3. Render the HTML with the data
        # 'roadmap' will be available in your HTML as {{ roadmap.title }}, etc.
        html_content = template.render(roadmap=roadmap)

        # 4. Generate the PDF
        # base_url is critical—it tells WeasyPrint where to look for 
        # local assets like CSS files or images referenced in the HTML.
        HTML(string=html_content, base_url=current_dir).write_pdf(target=output_path)
        
        logger.info(f"Successfully generated PDF at {output_path}")

    except Exception as e:
        logger.error(f"WeasyPrint/Jinja2 Error: {str(e)}")
        raise PDFRenderError(f"Internal PDF Rendering Error: {str(e)}")