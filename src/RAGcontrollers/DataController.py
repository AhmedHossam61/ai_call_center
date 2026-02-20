import os
import re
import uuid
import time
import shutil
import unicodedata
from fastapi import UploadFile, HTTPException
from helpers.config import get_settings

class DataController:
    
    def __init__(self):
        self.app_settings = get_settings()
        
        # Path: AI-CALL-CENTER/src/assets/files
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.files_dir = os.path.join(self.base_dir, "src", "assets", "files")

        # Ensure directory exists
        os.makedirs(self.files_dir, exist_ok=True)

        # Constraints
        self.max_file_size = 10 * 1024 * 1024  # 10MB limit
        self.allowed_extensions = {'.pdf', '.docx', '.txt', '.csv'}

    def validate_file(self, file: UploadFile):
        """Checks file extension and size before processing."""
        # Check Extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in self.allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type {ext} not allowed.")

        # Check Size (Note: some servers require seeking to check size)
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)  # Reset pointer to the start
        
        if file_size > self.max_file_size:
            raise HTTPException(status_code=413, detail="File too large. Max limit is 10MB.")
        
        return True

    def clean_filename(self, filename: str) -> str:
        """Sanitizes filename for system compatibility."""
        base_name, extension = os.path.splitext(filename)
        # Normalize and remove non-ascii
        clean_name = unicodedata.normalize('NFKD', base_name).encode('ascii', 'ignore').decode('ascii')
        # Replace special chars with underscore
        clean_name = re.sub(r'[^\w\s-]', '', clean_name).strip().lower()
        clean_name = re.sub(r'[-\s]+', '_', clean_name)
        
        # Add unique ID to prevent collisions
        unique_id = uuid.uuid4().hex[:8]
        timestamp = int(time.time())
        
        return f"{timestamp}_{unique_id}_{clean_name}{extension.lower()}"

    def clean_content(self, file_path: str):
        """
        Basic content cleaning: Removes excessive newlines and 
        normalizes encoding (mainly for .txt files).
        For PDF/Docx, this logic is usually handled later by Docling/Parsers.
        """
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Remove redundant whitespaces/newlines
            cleaned_content = re.sub(r'\n\s*\n', '\n', content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

    async def save_file(self, file: UploadFile):
        """Workflow: Validate -> Clean Name -> Save -> Clean Content."""
        self.validate_file(file)
        
        new_filename = self.clean_filename(file.filename)
        final_path = os.path.join(self.files_dir, new_filename)

        try:
            with open(final_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Post-save content cleaning for text files
            self.clean_content(final_path)
            
            return {
                "status": "success",
                "filename": new_filename,
                "path": final_path
            }
        except Exception as e:
            if os.path.exists(final_path): os.remove(final_path)
            raise HTTPException(status_code=500, detail=str(e))

    def load_file_metadata(self):
        """Loads a list of all validated files in the folder."""
        return [
            {"name": f, "path": os.path.join(self.files_dir, f)}
            for f in os.listdir(self.files_dir)
            if os.path.isfile(os.path.join(self.files_dir, f))
        ]
    
   