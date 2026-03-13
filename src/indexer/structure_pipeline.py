from indexer.pdf_loader import PDFLoader
from indexer.regulation_parser import RegulationParser


class StructurePipeline:

    def __init__(self):

        self.loader = PDFLoader()

        self.parser = RegulationParser()

    def run(self, pdf_path):

        pages = self.loader.load(pdf_path)

        estrutura = self.parser.parse(pages)

        return estrutura