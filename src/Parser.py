import pdfplumber
from llama_parse import LlamaParse

class PdfParser():
    def __init__(self,llamaparse_api_key):
        self.llamaparse_api_key = llamaparse_api_key

    def detect_tables_in_pdf(self,pdf_path):
        pages_with_tables = []
        pages_without_tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                if page.find_tables():
                    pages_with_tables.append(page_num)
                else:
                    pages_without_tables.append(page_num)
        return pages_with_tables,pages_without_tables

    def extract_text_using_pdfplumber(self,pdf_path,pages_without_tables):
        extracted_pages = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in pages_without_tables:
                if 1 <= page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]  # pdfplumber uses 0-indexing
                    text = page.extract_text()
                    if text:
                        extracted_pages[page_num] = text.strip()
                else:
                    print(f"Warning: Page {page_num} is out of range and will be skipped.")
        return extracted_pages

    def extract_mds_using_llamaparse(self,pdf_path, pages_with_tables):
        extracted_pages = {}
        target_pages = ",".join([str(i-1) for i in pages_with_tables])
        # parsing_instruction = "Extract table headers and link each cell to its corresponding header. Apply any specified instructions provided for the table (e.g., '000s omitted', 'values in crore rupees') to the table values where applicable. Capture numerical values (e.g., currency, percentages) and dates in a standard format."
        if len(pages_with_tables):
            parser = LlamaParse(api_key=self.llamaparse_api_key,target_pages=target_pages,result_type="markdown",verbose=False)
            page_markdowns = parser.load_data(pdf_path)
            for page_num,item in zip(pages_with_tables,page_markdowns):
                extracted_pages[page_num] = item
        return extracted_pages

    def create_md_file(self,extracted_pages,filename):
        with open(filename, 'w', encoding="utf-8") as md_file:
            for page_num,page in extracted_pages.items():
                md_file.write(page.text)
                md_file.write("\n\n---\n\n")

    def __call__(self,pdf_path):
        pages_with_tables, pages_without_tables = self.detect_tables_in_pdf(pdf_path)
        extracted_pages_pdfplumb = self.extract_text_using_pdfplumber(pdf_path,pages_without_tables)
        # create_md_file(extracted_pages_llamaparse,"parsed_file_with_instruction")
        extracted_pages_llamaparse = self.extract_mds_using_llamaparse(pdf_path,pages_with_tables)
        return extracted_pages_pdfplumb, extracted_pages_llamaparse

    

    