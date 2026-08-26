# 🐱 MeowVision AI — 66 Cat Breeds Classifier Studio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cat-breed-classifier-pykoqmu7mjjmnjmzkpk9wr.streamlit.app/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Architecture-ConvNeXt--Tiny-00599C)
![Classes](https://img.shields.io/badge/Classes-66%20Breeds-brightgreen)
![Top-5 Accuracy](https://img.shields.io/badge/Val%20Top--5%20Acc-87.34%25-blueviolet)

> **Live Demo:** [https://cat-breed-classifier-pykoqmu7mjjmnjmzkpk9wr.streamlit.app/](https://cat-breed-classifier-pykoqmu7mjjmnjmzkpk9wr.streamlit.app/)

**MeowVision AI** คือระบบปัญญาประดิษฐ์เชิงลึก (Deep Computer Vision Web Application) สำหรับจำแนกสายพันธุ์แมวสากลจำนวน 66 สายพันธุ์ พัฒนาขึ้นโดยต่อยอดจากชุดข้อมูลที่ผ่านกระบวนการ **Image Data Cleaning & Preprocessing Pipeline** และนำมาฝึกสอนด้วยโมเดลสถาปัตยกรรมระดับ State-of-the-Art **ConvNeXt-Tiny** ร่วมกับเทคนิค **CutMix Augmentation** พร้อมระบบตรวจสอบคุณภาพภาพ (Quality Auditing) ก่อนวิเคราะห์ผลจริงบนหน้าเว็บ[cite: 2]

---

## 🌟 จุดเด่นและฟังก์ชันการทำงาน (Key Features)

* **Multi-Input Scanning:** รองรับการอัปโหลดไฟล์ภาพนิ่ง (JPG, JPEG, PNG) และการสแกนสดผ่านกล้อง Webcam/Smartphone
* **Real-time Image Quality Audit:** ตรวจสอบความเบลอด้วย Laplacian Variance ($\text{Var}(\Delta I) \ge 100$) และวัดค่าความสว่างเฉลี่ยของภาพก่อนส่งเข้าโมเดล[cite: 2]
* **Top-5 Confidence Breakdown:** แสดงสายพันธุ์ที่ตรงที่สุดพร้อมป้ายความมั่นใจ และแจกแจงความน่าจะเป็น 5 อันดับแรก (Top-5 Probabilities)
* **Breed Encyclopedia Integration:** ดึงข้อมูลถิ่นกำเนิด ลักษณะนิสัย และเอกลักษณ์ประจำสายพันธุ์มาแสดงอัตโนมัติ
* **Optimized for Cloud Deployment:** โมเดลผ่านการบีบอัดค่าน้ำหนักด้วย **FP16 Half-Precision Compression** เหลือเพียง ~57 MB ทำให้โหลดประมวลผลบน Cloud ได้รวดเร็ว

---
👥 ผู้จัดทำและข้อมูลโครงงานโครงงานนี้เป็นส่วนหนึ่งของวิชาประมวลผลภาพดิจิทัลและการเตรียมข้อมูล 
(Image Dataset Cleaning & Machine Learning Pipeline):  
หัวข้อโครงงาน: End-to-End Image Cleaning Pipeline to Cat Breed Classification Web Applicationช
Repository [ต้นทาง ](https://github.com/yuulululu/Data-Cleaning-Preprocessing-Pipeline-)
Pipeline: Data-Cleaning-Preprocessing-PipelineLive 
Web App: [MeowVision AI on Streamlit Cloud](https://cat-breed-classifier-pykoqmu7mjjmnjmzkpk9wr.streamlit.app/)
---
## 🏗️ สถาปัตยกรรมและกระบวนการพัฒนา (End-to-End Pipeline)

```mermaid
flowchart LR
    A["Raw Dataset<br>(Kaggle API)"] --> B["Data Cleaning & EDA<br>(Laplacian + pHash)"]
    B --> C["Stratified Group Split<br>(No Data Leakage)"]
    C --> D["ConvNeXt-Tiny Training<br>(CutMix + Label Smoothing)"]
    D --> E["FP16 Compression<br>(~57 MB)"]
    E --> F["Streamlit Cloud<br>Deployment"]
