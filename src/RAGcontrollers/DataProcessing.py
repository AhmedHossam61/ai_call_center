import os
import uuid
from docling.document_converter import DocumentConverter
import semchunk
from .DataController import DataController
import traceback

class DataProcessing:
    def __init__(self):
        # Access file paths and metadata
        self.data_controller = DataController()
        
        # Docling for high-fidelity PDF/DOCX parsing
        self.converter = DocumentConverter()
        
        # semchunk for meaning-aware splitting (word-level token counter)
        self._chunk_size = 800
        self._token_counter = lambda t: len(t.split())

    def process_single_file(self, file_path: str, file_name: str):
        """
        Processes one specific file. Used by the FastAPI upload endpoint.
        """
        print(f"\n{'='*60}")
        print(f"📄 Processing: {file_name}")
        print(f"📁 Path: {file_path}")
        print(f"📊 Size: {os.path.getsize(file_path) / 1024:.2f} KB")
        print(f"{'='*60}")
        
        markdown_content = None
        
        try:
            # 1. Parsing Stage (Docling)
            print("🔄 Trying Docling conversion...")
            result = self.converter.convert(file_path)
            markdown_content = result.document.export_to_markdown()
            print(f"✅ Docling SUCCESS - {len(markdown_content)} chars extracted")
            
        except Exception as e:
            # Docling may fail due to transformers incompatibility (RTDetrImageProcessor).
            print(f"❌ Docling FAILED: {type(e).__name__}: {str(e)[:100]}")
            
            # Fallback: for PDFs, try pdfplumber to extract plain text
            if file_path.lower().endswith('.pdf'):
                try:
                    print("🔄 Trying pdfplumber fallback...")
                    import pdfplumber
                    pages = []
                    with pdfplumber.open(file_path) as pdf:
                        print(f"📖 PDF has {len(pdf.pages)} page(s)")
                        for page_num, p in enumerate(pdf.pages, 1):
                            page_text = p.extract_text()
                            if page_text:
                                pages.append(page_text)
                                print(f"   ✓ Page {page_num}: {len(page_text)} chars")
                            else:
                                print(f"   ✗ Page {page_num}: NO TEXT (image-based?)")
                    
                    if pages:
                        markdown_content = "\n\n".join(pages)
                        print(f"✅ pdfplumber SUCCESS - {len(markdown_content)} total chars")
                    else:
                        print(f"❌ FAILED: No text in any page - PDF is image-based or empty")
                        print(f"{'='*60}\n")
                        return []
                        
                except Exception as e2:
                    print(f"❌ pdfplumber FAILED: {type(e2).__name__}: {str(e2)[:100]}")
                    traceback.print_exc()
                    print(f"{'='*60}\n")
                    return []
            else:
                print(f"❌ Not a PDF - no fallback available")
                print(f"{'='*60}\n")
                return []
        
        # Validate content
        if not markdown_content or len(markdown_content.strip()) == 0:
            print(f"❌ FAILED: Extracted content is empty")
            print(f"{'='*60}\n")
            return []
        
        try:
            # 2. Chunking Stage — semantic chunking via semchunk
            print(f"🔄 Chunking {len(markdown_content)} chars (semchunk, size={self._chunk_size})...")
            chunks = semchunk.chunk(
                text=markdown_content,
                chunk_size=self._chunk_size,
                token_counter=self._token_counter,
            )
            print(f"✅ Created {len(chunks)} chunks")
            
            if len(chunks) == 0:
                print(f"❌ FAILED: Chunking produced 0 chunks!")
                print(f"{'='*60}\n")
                return []
            
            # 3. Metadata Attachment
            processed_chunks = []
            for i, chunk_text in enumerate(chunks):
                processed_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "metadata": {
                        "source_file": file_name,
                        "chunk_index": i,
                        "file_path": file_path,
                        "content_type": "markdown"
                    }
                })
            
            print(f"✅ SUCCESS: {len(processed_chunks)} chunks ready for indexing")
            print(f"{'='*60}\n")
            return processed_chunks
            
        except Exception as e:
            print(f"❌ Chunking FAILED: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            print(f"{'='*60}\n")
            return []

    def process_all_files(self):
        """
        Iterates through all files managed by the DataController.
        """
        all_processed_data = []
        files_metadata = self.data_controller.load_file_metadata()
        
        for file_info in files_metadata:
            print(f"Processing: {file_info['name']}...")
            
            # Use the single-file method to keep logic consistent
            file_chunks = self.process_single_file(file_info['path'], file_info['name'])
            all_processed_data.extend(file_chunks)
                
        return all_processed_data