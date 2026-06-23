"""
Dataset Generator for Social Media Disinformation Detection
Generates a simulated dataset of real and fake social media posts.
"""
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import hashlib
from datetime import datetime

random.seed(42)
np.random.seed(42)

REAL_POSTS = [
    {
        "text": "Federal University of Technology Minna ranked among top 10 universities in Nigeria for cyber security research. Congratulations to all faculty members and students.",
        "category": "education"
    },
    {
        "text": "Breaking: New solar power plant inaugurated in Niger State, providing electricity to over 50,000 households. A major step towards sustainable energy development.",
        "category": "technology"
    },
    {
        "text": "Nigeria's GDP grows by 3.2% in Q1 2026, exceeding analysts expectations. The agricultural and technology sectors led the growth according to NBS data.",
        "category": "economy"
    },
    {
        "text": "Super Eagles defeat Ghana 2-1 in AFCON qualifier at Moshood Abiola Stadium. Goals from Osimhen and Lookman secure victory for Nigeria.",
        "category": "sports"
    },
    {
        "text": "CBN announces new monetary policy aimed at stabilizing the naira. Interest rates adjusted to combat inflation while supporting economic growth.",
        "category": "economy"
    },
    {
        "text": "Nigerian tech startup raises $15 million in Series A funding. The company develops AI-powered solutions for agricultural monitoring across West Africa.",
        "category": "technology"
    },
    {
        "text": "UNICEF launches clean water initiative in rural communities across Northern Nigeria. Over 200 boreholes to be constructed in the first phase.",
        "category": "health"
    },
    {
        "text": "FUTMINNA students win national hackathon competition with their blockchain-based certificate verification system. Team of 5 computer science students.",
        "category": "education"
    },
    {
        "text": "Federal Government approves new minimum wage of N70,000 for civil servants. Implementation to begin from next month according to the labor ministry.",
        "category": "politics"
    },
    {
        "text": "Nigerian researchers develop low-cost water purification system using locally sourced materials. Published in international journal of environmental science.",
        "category": "technology"
    },
    {
        "text": "Lagos State introduces electric bus system for public transportation. First batch of 50 buses to operate on major routes across the state.",
        "category": "technology"
    },
    {
        "text": "National Youth Service Corps mobilizes 350,000 graduates for the 2026 batch A orientation program. Orientation camps open across all 36 states.",
        "category": "education"
    },
    {
        "text": "NNPC reports record crude oil production of 1.8 million barrels per day. Revenue expected to boost foreign exchange reserves significantly.",
        "category": "economy"
    },
    {
        "text": "Nigerian women football team qualifies for FIFA Women World Cup after defeating South Africa 3-0 in the final qualifier.",
        "category": "sports"
    },
    {
        "text": "New cybersecurity framework launched by NITDA to protect critical national infrastructure. Framework aligns with international standards and best practices.",
        "category": "technology"
    },
    {
        "text": "Abuja Light Rail project reaches 85% completion. Federal Capital Territory minister confirms launch date set for September 2026.",
        "category": "infrastructure"
    },
    {
        "text": "Nigerian film industry generates $7.2 billion in revenue, making Nollywood the second largest film industry globally by number of films produced.",
        "category": "entertainment"
    },
    {
        "text": "Independent National Electoral Commission begins voter registration exercise for 2027 general elections. Online portal opens next week.",
        "category": "politics"
    },
    {
        "text": "Nigerian scientists discover new plant species in Cross River National Forest. Research team from University of Calabar publishes findings in Nature.",
        "category": "science"
    },
    {
        "text": "MTN Nigeria expands 5G network coverage to 12 additional states. Investment of $200 million allocated for infrastructure development.",
        "category": "technology"
    },
    {
        "text": "National Assembly passes Cybercrime Prevention Amendment Bill. New provisions include stricter penalties for ransomware attacks and data theft.",
        "category": "politics"
    },
    {
        "text": "Dangote Refinery achieves full operational capacity of 650,000 barrels per day. Domestic fuel prices expected to stabilize in coming months.",
        "category": "economy"
    },
    {
        "text": "Nigerian para-athletes win 8 medals at World Para Athletics Championship. Tobi Amusan sets new world record in 100m hurdles.",
        "category": "sports"
    },
    {
        "text": "Federal Ministry of Education launches digital literacy program for 10 million primary school students. Tablets preloaded with educational content distributed.",
        "category": "education"
    },
    {
        "text": "WHO commends Nigeria for achieving 90% polio vaccination coverage. Country remains free of wild poliovirus for the fifth consecutive year.",
        "category": "health"
    },
]

FAKE_TEMPLATES = [
    {
        "template": "EXPOSED: {authority} has been secretly {action} for months. Sources confirm that {detail}. Share before this gets taken down!",
        "variations": [
            {"authority": "the Central Bank", "action": "printing counterfeit notes", "detail": "over N500 billion in fake currency has entered circulation"},
            {"authority": "the Federal Government", "action": "selling national assets to foreign companies", "detail": "NNPC, Dangote Refinery, and major highways were sold for fractions of their value"},
            {"authority": "NITDA", "action": "monitoring all citizen communications", "detail": "a secret surveillance program has been tracking millions of Nigerians"},
            {"authority": "the Ministry of Health", "action": "hiding vaccine side effects", "detail": "internal documents reveal over 10,000 adverse reactions were covered up"},
        ]
    },
    {
        "template": "URGENT ALERT: {event}! {consequence}. Do NOT {action}. Forward to everyone you know!",
        "variations": [
            {"event": "New Naira notes contain tracking chips", "consequence": "The government can now trace every transaction you make", "action": "use the new currency"},
            {"event": "Foreign hackers have compromised NIBSS", "consequence": "All bank accounts in Nigeria are at risk of being emptied", "action": "perform any online banking"},
            {"event": "Chemical clouds approaching from the Sahara", "consequence": "Toxic air will cover Northern Nigeria within 48 hours", "action": "go outside without protection"},
            {"event": "UNESCO declares Nigerian universities不合格", "consequence": "All degrees from Nigerian institutions will be invalidated internationally", "action": "trust any official communication"},
        ]
    },
    {
        "template": "BREAKING: {entity} announces {announcement}. This will {impact}. Multiple sources confirm this is happening.",
        "variations": [
            {"entity": "The Nigerian Military", "announcement": "immediate implementation of 24-hour curfew in 12 states", "impact": "affect over 50 million citizens starting midnight"},
            {"entity": "The World Bank", "announcement": "suspension of all loans to Nigeria", "impact": "cripple infrastructure projects worth $30 billion"},
            {"entity": "Google Nigeria", "announcement": "shutdown of all services in the country", "impact": "millions of businesses lose access to Gmail, YouTube, and cloud services"},
            {"entity": "The Supreme Court", "announcement": "invalidation of the 2023 presidential election results", "impact": "trigger a constitutional crisis and new elections within 90 days"},
        ]
    },
    {
        "template": "EXCLUSIVE: Leaked documents show {organization} planned {scheme}. {evidence}. This is the biggest scandal in Nigeria's history!",
        "variations": [
            {"organization": "INEC", "scheme": "to rig the upcoming elections using modified BVAS machines", "evidence": "Technical specifications show backdoor access for vote manipulation"},
            {"organization": "the NNPC", "scheme": "to divert $4.8 billion in oil revenue to offshore accounts", "evidence": "Bank records and transfer authorizations leaked by anonymous insiders"},
            {"organization": "the EFCC", "scheme": "to selectively prosecute political opponents while protecting allies", "evidence": "Internal memo reveals a list of protected individuals and target list"},
            {"organization": "the Nigerian Army", "scheme": "to stage attacks for budget justification", "evidence": "Communications between senior officers reveal coordinated false flag operations"},
        ]
    },
    {
        "template": "WARNING: {product} has been found to {harm}. {authority} is hiding this information. {instruction} immediately!",
        "variations": [
            {"product": "Popular baby formula brands sold in Nigeria", "harm": "contain dangerous levels of lead and mercury", "authority": "NAFDAC", "instruction": "Stop feeding your children these products"},
            {"product": "The new 5G masts being installed nationwide", "harm": "emit radiation that causes brain tumors and infertility", "authority": "NCC", "instruction": "Move your family away from any 5G tower"},
            {"product": "Bottled water from major Nigerian brands", "harm": "contain microscopic tracking devices", "authority": "SON", "instruction": "Switch to sachet water immediately"},
            {"product": "Imported rice sold in Nigerian markets", "harm": "is genetically modified to cause permanent infertility", "authority": "NAFDAC", "instruction": "Only buy rice from verified local farmers"},
        ]
    },
]


def generate_image(
    text: str,
    is_fake: bool,
    category: str = "general",
    size: Tuple[int, int] = (640, 480)
) -> Image.Image:
    """
    Generate a synthetic social media post image.
    Real posts have natural-looking backgrounds, fake posts have subtle artifacts.
    """
    if is_fake:
        base_color = (
            random.randint(20, 80),
            random.randint(20, 80),
            random.randint(120, 200)
        )
    else:
        base_color = (
            random.randint(180, 240),
            random.randint(180, 240),
            random.randint(180, 240)
        )
    
    image = Image.new("RGB", size, base_color)
    draw = ImageDraw.Draw(image)
    
    for _ in range(random.randint(20, 50)):
        cx = random.randint(0, size[0])
        cy = random.randint(0, size[1])
        rx = random.randint(10, 80)
        ry = random.randint(10, 80)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color, outline=None)
    
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, size[0])
        y1 = random.randint(0, size[1])
        x2 = random.randint(0, size[0])
        y2 = random.randint(0, size[1])
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 3))
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    words = text.split()[:15]
    wrapped_text = " ".join(words) + "..."
    
    text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (size[0] - text_width) // 2
    text_y = size[1] - text_height - 30
    
    draw.rectangle(
        [text_x - 10, text_y - 5, text_x + text_width + 10, text_y + text_height + 5],
        fill=(0, 0, 0, 128)
    )
    draw.text((text_x, text_y), wrapped_text, fill="white", font=font)
    
    if is_fake:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return image


def generate_dataset(
    num_real: int = 25,
    num_fake: int = 25,
    output_dir: str = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate a simulated dataset of real and fake social media posts.
    
    Returns:
        Tuple of (real_posts, fake_posts) dictionaries
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        output_dir = Path(output_dir)
    
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    real_posts = []
    fake_posts = []
    
    for i in range(num_real):
        post = REAL_POSTS[i % len(REAL_POSTS)]
        
        image = generate_image(
            post["text"],
            is_fake=False,
            category=post["category"]
        )
        
        image_filename = f"real_{i:03d}.png"
        image_path = real_dir / image_filename
        image.save(image_path, "PNG")
        
        real_posts.append({
            "id": f"real_{i:03d}",
            "text": post["text"],
            "image_path": str(image_path),
            "label": 0,
            "category": post["category"],
            "is_fake": False
        })
    
    for i in range(num_fake):
        template_group = random.choice(FAKE_TEMPLATES)
        variation = random.choice(template_group["variations"])
        fake_text = template_group["template"].format(**variation)
        
        image = generate_image(
            fake_text,
            is_fake=True,
            category="disinformation"
        )
        
        image_filename = f"fake_{i:03d}.png"
        image_path = fake_dir / image_filename
        image.save(image_path, "PNG")
        
        fake_posts.append({
            "id": f"fake_{i:03d}",
            "text": fake_text,
            "image_path": str(image_path),
            "label": 1,
            "category": "disinformation",
            "is_fake": True
        })
    
    all_posts = real_posts + fake_posts
    random.shuffle(all_posts)
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "total_samples": len(all_posts),
            "real_samples": len(real_posts),
            "fake_samples": len(fake_posts),
            "generated_at": datetime.now().isoformat(),
            "posts": all_posts
        }, f, indent=2)
    
    print(f"Dataset generated successfully!")
    print(f"  Real posts: {len(real_posts)}")
    print(f"  Fake posts: {len(fake_posts)}")
    print(f"  Total: {len(all_posts)}")
    print(f"  Saved to: {output_dir}")
    
    return real_posts, fake_posts


def load_dataset(data_dir: str = None) -> List[Dict]:
    """Load the generated dataset from metadata.json."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    
    metadata_path = data_dir / "metadata.json"
    
    if not metadata_path.exists():
        print("Dataset not found. Generating...")
        generate_dataset(output_dir=data_dir)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return metadata["posts"]


def split_dataset(
    posts: List[Dict],
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train, validation, and test sets.
    """
    random.seed(seed)
    
    real_posts = [p for p in posts if p["label"] == 0]
    fake_posts = [p for p in posts if p["label"] == 1]
    
    random.shuffle(real_posts)
    random.shuffle(fake_posts)
    
    n_real_test = int(len(real_posts) * test_size)
    n_real_val = int(len(real_posts) * val_size)
    n_fake_test = int(len(fake_posts) * test_size)
    n_fake_val = int(len(fake_posts) * val_size)
    
    test_set = real_posts[:n_real_test] + fake_posts[:n_fake_test]
    val_set = real_posts[n_real_test:n_real_test + n_real_val] + fake_posts[n_fake_test:n_fake_test + n_fake_val]
    train_set = real_posts[n_real_test + n_real_val:] + fake_posts[n_fake_test + n_fake_val:]
    
    random.shuffle(test_set)
    random.shuffle(val_set)
    random.shuffle(train_set)
    
    return train_set, val_set, test_set


if __name__ == "__main__":
    real_posts, fake_posts = generate_dataset()
    all_posts = real_posts + fake_posts
    train, val, test = split_dataset(all_posts)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train)} samples")
    print(f"  Validation: {len(val)} samples")
    print(f"  Test: {len(test)} samples")
