import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st

# 1. ตั้งค่าหน้าเพจ
st.set_page_config(
    page_title="MeowScan 🐾 แยกสายพันธุ์น้องแมว",
    page_icon="🐱",
    layout="centered"
)

# 2. ปรับแต่ง CSS ธีมน่ารัก พาสเทล ละมุนตา
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mitr:wght@300;400;500;600&family=Nunito:wght@400;700&display=swap');
    
    * {
        font-family: 'Mitr', 'Nunito', sans-serif;
    }
    
    /* สีพื้นหลังพาสเทล */
    .stApp {
        background: linear-gradient(135deg, #FFF5F7 0%, #F3F0FF 50%, #E8F5E9 100%);
    }
    
    /* การ์ดหัวเรื่อง */
    .hero-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 24px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(255, 182, 193, 0.3);
        border: 2px solid #FFE4E8;
        margin-bottom: 25px;
    }
    
    .hero-title {
        color: #D84374;
        font-size: 2.3rem;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .hero-subtitle {
        color: #7D6B7D;
        font-size: 1.05rem;
    }
    
    /* กล่องผลลัพธ์ */
    .result-box {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 20px;
        border: 2px dashed #FFB6C1;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
        margin-top: 15px;
    }
    
    .winner-breed {
        color: #C2185B;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .confidence-badge {
        background: #FFE8EC;
        color: #E91E63;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 3. โหลด Model & Classes
@st.cache_resource
def load_cat_model():
    if not os.path.exists('class_names.json') or not os.path.exists('cat_breed_model.pth'):
        return None, None
    
    with open('class_names.json', 'r', encoding='utf-8') as f:
        class_names = json.load(f)
        
    num_classes = len(class_names)
    model = models.mobilenet_v3_small()
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    model.load_state_dict(torch.load('cat_breed_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, class_names

model, class_names = load_cat_model()

# Image Preprocessing สำหรับการ Predict
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ข้อมูลนิสัย/คำอธิบายเสริมสำหรับแต่ละสายพันธุ์ (ปรับเพิ่มได้)
CAT_INFO = {
    "Bengal": "🐯 เบงกอล: ปราดเปรียว ลายเสือดาว ขี้เล่น รักการผจญภัย",
    "British Shorthair": "🧸 บริติช ช็อตแฮร์: นิ่งสุขุม แก้มกลมแน่น ขี้อ้อนแบบเงียบๆ",
    "Persian": "👑 เปอร์เซีย: ขนฟูสลวย เรียบร้อย อ่อนหวาน รักความสงบ",
    "Ragdoll": "🎀 แร็กดอลล์: ตัวนุ่มนิ่ม ยอมให้อุ้ม ตาฟ้า ตาโต อ่อนโยนมาก",
    "Siamese": "🐾 วิเชียรมาศ: แมวไทยโบราณ ฉลาด ช่างคุย ช่างประจบ",
    "Sphynx": "👽 สฟิงซ์: ไร้ขนแต่ตัวอุ่นมาก ขี้อ้อน ชอบเกาะติดเจ้าของ",
    "Maine Coon": "🦁 เมนคูน: แมวยักษ์ใจดี หางพวงฟู เป็นมิตรกับทุกคน",
    "Scottish Fold": "🦉 สก็อตติช โฟลด์: หูพับสุดน่ารัก หน้ากลม ชอบนั่งท่าแปลกๆ"
}

# 4. ส่วนแสดงผล UI
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🐾 MeowScan AI 🐱</div>
    <div class="hero-subtitle">ระบบ AI ตรวจจับและวิเคราะห์สายพันธุ์แมวเหมียวสุดน่ารัก</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ ไม่พบไฟล์ `cat_breed_model.pth` หรือ `class_names.json`! กรุณารัน `python train_model.py` เพื่อเทรนโมเดลก่อนนะครับ")
else:
    uploaded_file = st.file_uploader(
        "✨ อัปโหลดรูปภาพน้องแมว (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.markdown("### 📸 รูปน้องแมวของคุณ")
            st.image(image, use_container_width=True, caption="รูปภาพที่อัปโหลด")

        with col2:
            st.markdown("### 🔍 ผลการวิเคราะห์")
            with st.spinner("กำลังส่องความน่ารัก... ✨"):
                # Preprocess & Inference
                input_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                # ดึง Top 3
                top3_prob, top3_idx = torch.topk(probabilities, k=min(3, len(class_names)))
                
                top_breed = class_names[top3_idx[0].item()]
                top_conf = top3_prob[0].item() * 100

                # แสดงผลการ์ดชนะเลิศ
                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size: 0.95rem; color: #888;">AI มั่นใจว่าเป็นพันธุ์:</div>
                    <div class="winner-breed">🐾 {top_breed}</div>
                    <div class="confidence-badge">ความแม่นยำ {top_conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # แสดงเกร็ดความรู้
                if top_breed in CAT_INFO:
                    st.info(CAT_INFO[top_breed])
                
                st.markdown("---")
                st.markdown("#### 📊 อันดับความน่าจะเป็น (Top 3)")
                for i in range(len(top3_idx)):
                    breed_name = class_names[top3_idx[i].item()]
                    prob_val = top3_prob[i].item() * 100
                    st.write(f"**{i+1}. {breed_name}** ({prob_val:.1f}%)")
                    st.progress(prob_val / 100.0)
                    
                st.balloons()