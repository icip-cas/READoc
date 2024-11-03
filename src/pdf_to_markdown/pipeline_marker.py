import pypdfium2 # Needs to be at the top to avoid warnings
from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models
from glob import glob
import os
import fire

configure_logging()

model_lst = load_all_models()

def transform(pdf_dir : str, md_dir : str):
    os.makedirs(md_dir, exist_ok=True)
    file_list = glob(f"{pdf_dir}/*.pdf")

    for pdf in file_list:
        md_text, images, out_meta = convert_single_pdf(pdf, model_lst, max_pages=None, langs=['English', 'Chinese'], batch_multiplier=3, start_page=None)

        with open(os.path.join(md_dir, pdf.split('/')[-1].replace('.pdf', '.md')), 'w') as f:
            f.write(md_text)

if __name__ == "__main__":
    fire.Fire(transform)