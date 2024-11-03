import pymupdf4llm
from glob import glob
import os
import fire

def transform(pdf_dir : str, md_dir : str):
    os.makedirs(md_dir, exist_ok=True)
    file_list = glob(f"{pdf_dir}/*.pdf")

    for pdf in file_list:
        md_text = pymupdf4llm.to_markdown(pdf)
        with open(os.path.join(md_dir, pdf.split('/')[-1].replace('.pdf', '.md')), 'w') as f:
            f.write(md_text)

if __name__ == "__main__":
    fire.Fire(transform)