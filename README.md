---
title: UngThuPhoi Lung CT Risk Demo
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# UngThuPhoi - Lung CT Risk Demo

He thong demo du doan nguy co ac tinh phoi tu anh CT va thong tin lam sang co ban.

Pipeline inference hien tai gom 2 nhanh model:

- Multimodal classification model: phan loai 3 muc nguy co `low_malignancy`, `medium_risk`, `high_malignancy`
- YOLO localization model: khoanh vung not/vung nghi ngo tren anh CT de ho tro giai thich truc quan

He thong chi phuc vu hoc tap va nghien cuu, khong thay the chan doan lam sang.

## Pipeline Hien Tai

1. Nguoi dung chon anh mau hoac upload anh CT phoi
2. Frontend kiem tra nhanh anh dau vao co phu hop mien CT phoi hay khong
3. Backend chay YOLO localization de tim vung not/vung nghi ngo
4. Backend chay model phan loai da phuong thuc tu anh ROI va thong tin benh nhan
5. Backend tra ve nhan du doan, xac suat tung lop, canh bao va anh da khoanh vung nghi ngo

## Model Production

Classification production:

```text
models/production/multimodal_model_3class_distilled_student.pth
```

Calibration:

```text
models/production/temperature_scaling_3class.json
```

YOLO localization production:

```text
models/localization/yolo_nodule_kaggle_m640_e80_fixray2_best.pt
```

YOLO runtime mac dinh:

```text
YOLO_IMAGE_SIZE=640
YOLO_CONF_THRESHOLD=0.12
```

## Dataset Chinh

Classification CSV production:

```text
dataset/multimodal_image_dataset_3class_roi.csv
```

Sample images duoc dung cho demo:

```text
data/processed/roi_images_3class/
data/processed/images/
frontend/public/sample-previews/
```

## Chay Local

### 1. Start backend FastAPI

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1
```

Backend chay tai:

```text
http://127.0.0.1:8000
```

### 2. Frontend React/Vite cho dev tuy chon

```powershell
cd frontend
npm install
npm.cmd run dev
```

Vite dev server mac dinh chay tai:

```text
http://127.0.0.1:5173
```

Frontend dev co the goi backend qua:

```text
VITE_API_BASE=http://127.0.0.1:8000
```

## API Chinh

```text
GET  /health
POST /predict
GET  /sample-image
```

## Hugging Face Spaces

Repo nay da duoc chuan bi theo huong Docker Space:

- `Dockerfile` build frontend React roi serve qua FastAPI
- `app_port` dat la `7860`
- frontend production mac dinh goi API theo same-origin
- khong can MongoDB vi he thong hien khong con luu lich su du doan

### Cach deploy

1. Tao Space moi tren Hugging Face va chon `Docker`
2. Push toan bo repo len Space
3. Cho Space build lai tu `Dockerfile`
4. Mo app tai URL cua Space sau khi build xong

Neu can build voi API base tuy chinh trong moi truong khac, co the truyen `VITE_API_BASE` trong qua trinh build Docker.
