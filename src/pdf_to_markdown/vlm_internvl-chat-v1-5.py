import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # use multiple GPUs to inference
import fitz 
from glob import glob

from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import fire


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


model_name = "OpenGVLab/InternVL-Chat-V1-5"
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True, 
    device_map='auto').eval() 

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


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
    for image in images:

        pixel_values = load_image(image, max_num=6).to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=2048,
            do_sample=True,
            top_p = 0.8,
            top_k = 100,
            temperature = 0.7,
            repetition_penalty = 1.05,
        )

        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        markdown += response.strip() + '\n\n'

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