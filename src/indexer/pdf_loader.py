import fitz


class PDFLoader:
    def load(self, file_path: str):
        pages = []

        with fitz.open(file_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text")
                extracted_tables = [
                    table.extract() for table in page.find_tables()
                ]
                pages.append({
                    "page": page_number,
                    "text": text,
                    "tables": extracted_tables,
                })

        return pages