# System Test Guide

File nay dung de kiem tra nhanh he thong production sau khi da bo MongoDB va lich su du doan.

## 1. Start Backend

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1
```

Backend chay tai:

```text
http://127.0.0.1:8000
```

Backend cung serve luon giao dien React da build tai:

```text
http://127.0.0.1:8000
```

## 2. Frontend Dev Tuy Chon

```powershell
cd frontend
npm install
npm.cmd run dev
```

Vite dev server chay tai:

```text
http://127.0.0.1:5173
```

## 3. Test Health

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Checklist pass:

```text
status = ok
model_loaded = true
checkpoint = multimodal_model_3class_distilled_student.pth
dataset_csv = dataset\multimodal_image_dataset_3class_roi.csv
yolo_localization_loaded = true
yolo_checkpoint = models\localization\yolo_nodule_kaggle_m640_e80_fixray2_best.pt
yolo_conf_threshold = 0.12
startup_error = null
yolo_startup_error = null
```

## 4. Test Predict Tren Giao Dien

1. Mo giao dien tai `http://127.0.0.1:8000`
2. Chon anh mau hoac upload anh CT phoi hop le
3. Nhap thong tin benh nhan
4. Bam du doan

Checklist pass:

- ket qua hien thi binh thuong
- anh vung nghi ngo duoc render neu YOLO phat hien box
- warning hien dung neu khong co box hoac thieu truong du lieu

## 5. Test API Predict Bang PowerShell

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/predict `
  -F "sample_path=data/processed/roi_images_3class/LIDC-IDRI-0184/LIDC-IDRI-0184_4268743d0705e877.png" `
  -F "localization_sample_path=data/processed/images/LIDC-IDRI-0184/df88dd43499edaf2.png" `
  -F "patient_name=Smoke Test" `
  -F "age=58" `
  -F "sex=female" `
  -F "smoking_status=never" `
  -F "tumor_size=16.2" `
  -F "family_history=no" `
  -F "symptom_score=2"
```

Checklist pass:

- response co `risk_level`
- response co `probabilities`
- response co `localization`
- response co `input_summary`

## 6. Test Frontend Build

```powershell
cd frontend
npm.cmd run build
```

Checklist pass:

- build thanh cong
- tao ra `frontend/dist`

## 7. Test Docker Space Readiness

Checklist:

- co `Dockerfile` o root project
- co `.dockerignore`
- `README.md` co YAML metadata voi `sdk: docker`
- frontend production mac dinh same-origin API
- khong con API lich su du doan va khong con MongoDB local
