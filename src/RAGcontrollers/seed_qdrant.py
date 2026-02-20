"""
Small helper to insert a sample document into Qdrant using the existing VectorDB class.
Run: python seed_qdrant.py
"""
import uuid
from RAGcontrollers.VectorDB import VectorDB


def main():
    db = VectorDB()
    sample_text = (
        "شركة المثال: تقدم خدمة الدعم الفني للسيارات والدراجات النارية. "
        "نحن نعمل من الأحد إلى الخميس ونتوفر عبر الهاتف والبريد الإلكتروني."
    )

    chunk = {
        'id': str(uuid.uuid4()),
        'text': sample_text,
        'metadata': {
            'db_id': 'sample-1',
            'source_file': 'seed_sample',
            'chunk_index': 0
        }
    }

    inserted = db.insert_chunks([chunk])
    print(f"Inserted {inserted} points into collection {db.settings.VECTOR_COLLECTION_NAME}")


if __name__ == '__main__':
    main()
