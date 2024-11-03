import os
from glob import glob
import base64
import requests
import fitz
from PIL import Image
from memory_tempfile import MemoryTempfile
import fire

tempfile = MemoryTempfile()

# OpenAI API Key
api_key = ""
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}
   
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def process_pdf(pdf_path, prompt):
    images = []
    pdf = fitz.open(pdf_path)
    page_num = pdf.page_count
    for id in range(page_num):
        page = pdf[id]
        page_rate = 3.0
        mat = fitz.Matrix(page_rate, page_rate)  
        pm = page.get_pixmap(matrix=mat, alpha=False)
        images.append(Image.frombytes("RGB", [pm.width, pm.height], pm.samples))

    markdown = ''
    with tempfile.TemporaryDirectory(dir=".") as td:
        for id, image in enumerate(images):
            image.save(f'{td}/{id}.jpg', 'JPEG', quality = 95)
            base64_image = encode_image(f'{td}/{id}.jpg')

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.8,
                }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            markdown += response.json()['choices'][0]['message']['content'].strip() + '\n\n'

    return markdown.strip()
       

def transform_with_prompt(pdf_dir : str, md_dir : str, prompt : str):
    os.makedirs(md_dir, exist_ok=True)
    file_list = glob(f"{pdf_dir}/*.pdf")

    for pdf in file_list:
        md_text = process_pdf(pdf, prompt)
        
        with open(os.path.join(md_dir, pdf.split('/')[-1].replace('.pdf', '.md')), 'w') as f:
            f.write(md_text)


if __name__ == "__main__":
    fire.Fire(transform_with_prompt)