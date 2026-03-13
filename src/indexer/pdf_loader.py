import fitz


class PDFLoader:

    def load(self, file_path: str):

        doc = fitz.open(file_path)

        pages = []

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")

            tables = page.find_tables()

            extracted_tables = []

            for table in tables:

                extracted_tables.append(table.extract())

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                    "tables": extracted_tables,
                }
            )

        return pages