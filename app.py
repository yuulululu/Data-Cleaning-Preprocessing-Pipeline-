import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps, ImageFilter
import streamlit as st

# =========================================================================
# 1. Page Configuration
# =========================================================================
st.set_page_config(
    page_title="MeowVision AI 🐾 ตรวจจับและวิเคราะห์สายพันธุ์แมว",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# 2. Advanced CSS & Animation Styling
# =========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mitr:wght@300;400;500;600;700&family=Nunito:wght@600;700;800&display=swap');
    
    * { font-family: 'Mitr', 'Nunito', sans-serif; }

    .stApp {
        background: radial-gradient(at 0% 0%, #FFF0F5 0px, transparent 50%),
                    radial-gradient(at 100% 0%, #F3E8FF 0px, transparent 50%),
                    radial-gradient(at 100% 100%, #E8F5E9 0px, transparent 50%),
                    radial-gradient(at 0% 100%, #FFF8E7 0px, transparent 50%),
                    #FDFBF7;
    }

    @keyframes floatAnimation {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-8px) rotate(2deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    .hero-container {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #FFD1DC;
        border-radius: 28px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(233, 30, 99, 0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 25px;
    }

    .hero-cat-icon {
        font-size: 3.2rem;
        display: inline-block;
        animation: floatAnimation 3s ease-in-out infinite;
    }

    .hero-title {
        color: #880E4F;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 5px 0;
    }

    .hero-tagline {
        color: #4A3E4D;
        font-size: 1.05rem;
    }

    .glass-card {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 22px;
        border: 1.5px solid #F0E6ED;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    .winner-card {
        background: linear-gradient(145deg, #FFFFFF 0%, #FFF5F7 100%);
        border: 2px solid #FF4081;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 12px 30px rgba(255, 64, 129, 0.15);
        margin-bottom: 20px;
    }

    .winner-badge {
        background: linear-gradient(135deg, #FF4081, #C2185B);
        color: #FFFFFF !important;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 50px;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .winner-name {
        color: #2D2438 !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 10px 0 5px 0;
    }

    .confidence-pill {
        background: #FCE4EC;
        color: #AD1457 !important;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 12px;
        border: 1px solid #F8BBD0;
    }

    .trait-box {
        background: #F8F9FA;
        border-left: 4px solid #FF4081;
        border-radius: 12px;
        padding: 14px 18px;
        color: #374151 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .audit-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }

    .audit-pass { background: #E8F5E9; color: #2E7D32; border: 1px solid #C8E6C9; }
    .audit-warn { background: #FFF3E0; color: #E65100; border: 1px solid #FFE0B2; }

    .rank-row {
        background: #FAFAFA;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 6px;
        border: 1px solid #EEEEEE;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 3. Model Loader & Quality Auditing
# =========================================================================
# ในไฟล์ app.py
@st.cache_resource
def load_cat_model():
    if not os.path.exists('class_names.json') or not os.path.exists('cat_breed_model.pth'):
        return None, None
    
    with open('class_names.json', 'r', encoding='utf-8') as f:
        class_names = json.load(f)
        
    num_classes = len(class_names)
    model = models.convnext_tiny()
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('cat_breed_model.pth', map_location=device)
    
    # แปลงค่าน้ำหนักกลับเป็น Float32 เพื่อความเสถียรบน CPU
    state_dict = {k: v.float() if torch.is_floating_point(v) else v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names

model, class_names = load_cat_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ฟังก์ชันตรวจสอบคุณภาพภาพ (Laplacian Filter & Brightness)
def audit_image(pil_img):
    gray = pil_img.convert('L')
    gray_np = np.array(gray, dtype=np.float32)
    
    # คำนวณ Laplacian Variance (หาความคมชัด)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray_np.shape
    if h > 3 and w > 3:
        padded = np.pad(gray_np, 1, mode='edge')
        conv = (
            padded[:-2, 1:-1] * laplacian[0, 1] +
            padded[1:-1, :-2] * laplacian[1, 0] +
            padded[1:-1, 1:-1] * laplacian[1, 1] +
            padded[1:-1, 2:] * laplacian[1, 2] +
            padded[2:, 1:-1] * laplacian[2, 1]
        )
        blur_var = float(conv.var())
    else:
        blur_var = 0.0

    mean_brightness = float(gray_np.mean())
    
    is_sharp = blur_var > 100.0
    is_good_light = 30.0 <= mean_brightness <= 230.0
    
    return {
        "blur_var": blur_var,
        "is_sharp": is_sharp,
        "brightness": mean_brightness,
        "is_good_light": is_good_light
    }

BREED_PROFILES = {
    "bengal": {"icon": "🐯", "origin": "สหรัฐอเมริกา", "trait": "ปราดเปรียว ว่องไว ลายขนเสือดาว รักการผจญภัยและชอบเล่นน้ำ"},
    "british_shorthair": {"icon": "🧸", "origin": "สหราชอาณาจักร", "trait": "สุขุม นิ่งสงบ แก้มกลมแน่น ขนแน่นนุ่ม อ่อนโยนและเป็นมิตร"},
    "persian": {"icon": "👑", "origin": "อิหร่าน", "trait": "ขนยาวฟูสลวย หน้าสั้น เรียบร้อย รักความสงบ ชื่นชอบการพักผ่อน"},
    "ragdoll": {"icon": "🎀", "origin": "สหรัฐอเมริกา", "trait": "ตาสีฟ้า ตัวใหญ่นุ่มนิ่ม เมื่ออุ้มจะทิ้งตัวนิ่ง อ่อนโยนและติดคนมาก"},
    "siamese": {"icon": "🐾", "origin": "ไทย (วิเชียรมาศ)", "trait": "แมวไทยโบราณ ลวดลายแต้มสีเข้ม 9 จุด ฉลาด ปราดเปรียว ช่างพูดคุย"},
    "sphynx": {"icon": "👽", "origin": "แคนาดา", "trait": "ไร้ขน ผิวหนังอุ่นนุ่ม ขี้อ้อน ชอบซุกไออุ่น มีพลังงานสูงและเป็นมิตร"},
    "maine_coon": {"icon": "🦁", "origin": "สหรัฐอเมริกา", "trait": "ยักษ์ใหญ่ใจดี หางฟูพวงหนา ขนกันน้ำ ทนทานต่อความหนาวเย็น"},
    "scottish_fold": {"icon": "🦉", "origin": "สกอตแลนด์", "trait": "หูพับเป็นเอกลักษณ์ ตาโตกลมบ๊อก ขี้เล่น ชอบนั่งท่าแปลกๆ"},
    "korat": {"icon": "💙", "origin": "ไทย (โคราช/สีสวาด)", "trait": "ขนสีเทาเงินประกาย ตาสีเขียวมรกต ฉลาด ความจำเป็นเลิศ มงคลนำโชค"},
    "khao_manee": {"icon": "💎", "origin": "ไทย (ขาวมณี)", "trait": "ขนขาวผุดผ่อง ตาสองสี (อัญมณี) เชื่อง อ่อนหวาน น่ารักน่าเอ็นดู"}
}

def get_breed_info(breed_slug):
    slug = breed_slug.lower().strip()
    if slug in BREED_PROFILES:
        return BREED_PROFILES[slug]
    return {
        "icon": "🐱",
        "origin": "สายพันธุ์แมวสากล",
        "trait": "มีเอกลักษณ์เฉพาะตัว สดใสร่าเริง และมีความเฉลียวฉลาดตามธรรมชาติของสายพันธุ์"
    }

# =========================================================================
# 4. Sidebar Controls
# =========================================================================
with st.sidebar:
    st.markdown("### 🐾 MeowVision System")
    st.markdown("""
    **Model:** ConvNeXt-Tiny (Modern Vision CNN)  
    **Dataset Classes:** 66 สายพันธุ์  
    **Pipeline:** Cleaned + Stratified Group Split
    """)
    st.markdown("---")
    st.markdown("### ⚙️ ตัวเลือกการแสดงผล")
    show_top5 = st.checkbox("แสดงผลวิเคราะห์ 5 อันดับแรก", value=True)
    show_audit = st.checkbox("ตรวจสอบคุณภาพภาพ (Quality Audit)", value=True)
    st.markdown("---")
    st.caption("🐱 Vocational Capstone Project | Computer Vision Web App")

# =========================================================================
# 5. Hero Header & Tabs Layout
# =========================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-cat-icon">🐱✨</div>
    <div class="hero-title">MeowVision AI Studio</div>
    <div class="hero-tagline">ระบบสแกน ตรวจสอบคุณภาพ และจำแนกสายพันธุ์แมว 66 สายพันธุ์ด้วย Modern Deep Vision</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ ไม่พบไฟล์ `cat_breed_model.pth` หรือ `class_names.json`! กรุณารัน `python train_model.py` ให้เสร็จก่อนนะครับ")
    st.stop()

tab_scan, tab_cam, tab_gallery, tab_pipeline = st.tabs([
    "📂 อัปโหลดรูปภาพ",
    "📷 สแกนผ่านกล้องสด",
    "🖼️ ภาพตัวอย่างทดสอบ",
    "📊 Pipeline & ข้อมูลโมเดล"
])

target_image = None

# --- TAB 1: อัปโหลดรูป ---
with tab_scan:
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพน้องแมว (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="uploader"
    )
    if uploaded_file:
        target_image = Image.open(uploaded_file).convert('RGB')

# --- TAB 2: ถ่ายผ่านกล้อง ---
with tab_cam:
    cam_file = st.camera_input("📸 ส่องกล้องไปที่หน้าน้องแมวแล้วกดถ่ายภาพ")
    if cam_file:
        target_image = Image.open(cam_file).convert('RGB')

# --- TAB 3: ภาพตัวอย่างทดสอบ ---
with tab_gallery:
    st.markdown("เลือกคลิกภาพตัวอย่างด้านล่างเพื่อทดสอบระบบได้ทันที:")
    val_base = "data/val"
    if os.path.exists(val_base):
        sample_breeds = [b for b in os.listdir(val_base) if os.path.isdir(os.path.join(val_base, b))][:4]
        cols = st.columns(4)
        for idx, breed in enumerate(sample_breeds):
            breed_dir = os.path.join(val_base, breed)
            files = [f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.png'))]
            if files:
                sample_img_path = os.path.join(breed_dir, files[0])
                sample_img = Image.open(sample_img_path).convert('RGB')
                with cols[idx]:
                    st.image(sample_img, caption=breed.replace('_', ' ').title(), use_container_width=True)
                    if st.button(f"ทดสอบ {breed.title()}", key=f"btn_{breed}"):
                        target_image = sample_img
    else:
        st.info("💡 สามารถทดสอบด้วยการอัปโหลดรูปภาพในแท็บแรกได้เลยครับ")

# --- TAB 4: Pipeline Insights ---
with tab_pipeline:
    st.markdown("### 🛠️ สถาปัตยกรรมระบบและ Data Cleaning Pipeline")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        **1. กระบวนการเตรียมข้อมูล (Preprocessing Pipeline):**
        * **Kaggle API Fetch:** โหลดข้อมูลแมวอัตโนมัติ ไม่เก็บรูปดิบใน Git
        * **Laplacian Filter Audit:** ตัดภาพที่เบลอ มีค่า Variance < 100
        * **Perceptual Hash:** ตรวจจับและลบภาพซ้ำ (Duplicate Images)
        * **Stratified Group Split:** แบ่ง Train/Val/Test (70/15/15) โดยป้องกัน Data Leakage
        """)
    with col_p2:
        st.markdown("""
        **2. สถาปัตยกรรมโมเดล (Model Specifications):**
        * **Backbone:** ConvNeXt-Tiny (ImageNet Pretrained)
        * **Regularization:** CutMix Augmentation + Label Smoothing (0.1)
        * **Optimizer:** AdamW + Cosine Annealing Decay
        * **Mixed Precision:** FP16 Acceleration บน RTX 3050 Ti
        """)

# =========================================================================
# 6. ผลการวิเคราะห์ (Inference & Display)
# =========================================================================
if target_image is not None:
    st.markdown("---")
    col_img, col_res = st.columns([1, 1.2], gap="large")

    with col_img:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📸 ภาพที่ส่งเข้าวิเคราะห์")
        st.image(target_image, use_container_width=True)
        
        # Image Quality Audit Breakdown
        if show_audit:
            audit = audit_image(target_image)
            st.markdown("#### 🔍 ผลการตรวจสอบคุณภาพภาพ (Quality Audit)")
            blur_class = "audit-pass" if audit['is_sharp'] else "audit-warn"
            blur_text = "คมชัด ผ่านเกณฑ์ ✅" if audit['is_sharp'] else "ภาพมีความเบลอ ⚠️"
            light_class = "audit-pass" if audit['is_good_light'] else "audit-warn"
            light_text = "แสงปกติ ผ่านเกณฑ์ ✅" if audit['is_good_light'] else "แสงมืด/สว่างเกินไป ⚠️"

            st.markdown(f"""
            <div>
                <span class="audit-badge {blur_class}">Laplacian Var: {audit['blur_var']:.1f} ({blur_text})</span><br><br>
                <span class="audit-badge {light_class}">Brightness: {audit['brightness']:.1f} ({light_text})</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        with st.spinner("🐾 AI กำลังวิเคราะห์ลักษณะสายพันธุ์..."):
            device = next(model.parameters()).device
            input_tensor = preprocess(target_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            top5_prob, top5_idx = torch.topk(probabilities, k=min(5, len(class_names)))
            top_raw = class_names[top5_idx[0].item()]
            top_breed = top_raw.replace('_', ' ').title()
            top_conf = top5_prob[0].item() * 100
            meta = get_breed_info(top_raw)

            # Winner Card
            st.markdown(f"""
            <div class="winner-card">
                <span class="winner-badge">✨ สายพันธุ์ที่ตรวจพบอันดับ 1</span>
                <div class="winner-name">{meta['icon']} {top_breed}</div>
                <div class="confidence-pill">ความมั่นใจ {top_conf:.1f}%</div>
                <div class="trait-box">
                    <strong>📍 ถิ่นกำเนิด:</strong> {meta['origin']}<br>
                    <strong>💡 บุคลิกและเอกลักษณ์:</strong> {meta['trait']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Ranking Breakdown
            if show_top5:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #2D2438; margin-top:0;'>📊 5 อันดับความเป็นไปได้สูงสุด (Top 5)</h4>", unsafe_allow_html=True)
                for i in range(len(top5_idx)):
                    r_name = class_names[top5_idx[i].item()]
                    c_name = r_name.replace('_', ' ').title()
                    p_val = top5_prob[i].item() * 100
                    icon = get_breed_info(r_name)['icon']

                    st.markdown(f"""
                    <div class="rank-row">
                        <span style="font-weight: 600; color: #333;">#{i+1} {icon} {c_name}</span>
                        <span style="font-weight: 700; color: #C2185B;">{p_val:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(p_val / 100.0)
                st.markdown("</div>", unsafe_allow_html=True)

            if top_conf > 50.0:
                st.balloons()