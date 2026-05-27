# UngThuPhoi - Lung CT Risk Demo

Hệ thống demo dự đoán nguy cơ ác tính phổi từ ảnh CT và thông tin lâm sàng cơ bản. Dự án hiện dùng hai nhánh model trong pipeline inference:

- Multimodal classification model: phân loại nguy cơ 3 lớp `low_malignancy`, `medium_risk`, `high_malignancy`.
- YOLO localization model: khoanh vùng nốt/vùng nghi ngờ trên ảnh CT để hỗ trợ giải thích trực quan.

Hệ thống chỉ phục vụ học tập/nghiên cứu, không thay thế chẩn đoán lâm sàng.

## Pipeline Hiện Tại

1. Người dùng chọn ảnh mẫu hoặc upload ảnh CT phổi.
2. Frontend kiểm tra nhanh ảnh đầu vào có phù hợp miền CT phổi hay không.
3. Backend chạy YOLO localization để tìm vùng nốt/vùng nghi ngờ.
4. Backend chạy model phân loại đa phương thức từ ảnh ROI và thông tin bệnh nhân.
5. Backend trả về nhãn dự đoán, xác suất từng lớp, confidence, cảnh báo nếu YOLO không phát hiện box nhưng classification nghiêng nguy cơ cao.
6. Nếu MongoDB đang chạy, kết quả dự đoán được lưu vào lịch sử.

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

Cấu hình YOLO runtime mặc định:

```text
YOLO_IMAGE_SIZE=640
YOLO_CONF_THRESHOLD=0.12
```

## Dataset Chính

Classification CSV production:

```text
dataset/multimodal_image_dataset_3class_roi.csv
```

Các manifest/index giữ để tham chiếu split và dữ liệu:

```text
dataset/dataset_index.csv
dataset/series_labels.csv
```

Ảnh production đang dùng:

```text
data/processed/roi_images_3class/
data/processed/images/
data/processed/full_preview_3class/
data/localization/yolo_nodule/
```

## Cấu Trúc Production Rút Gọn

```text
UngThuPhoi/
  backend/
    api_server.py
    app_demo.py
    __init__.py
  frontend/
    src/
    public/sample-previews/
    package.json
    .env
  dataset/
    multimodal_image_dataset_3class_roi.csv
    dataset_index.csv
    series_labels.csv
  data/
    processed/roi_images_3class/
    processed/images/
    processed/full_preview_3class/
    localization/yolo_nodule/
  models/
    production/
      multimodal_model_3class_distilled_student.pth
      temperature_scaling_3class.json
    localization/
      yolo_nodule_kaggle_m640_e80_fixray2_best.pt
      yolo_nodule_py311_e8_img416/weights/best.pt
  reports/
    final/
      multiclass_3class_eval_report_distilled_student.json
      production_patient_level_stability.json
    localization/
      yolo_production_detection_details_conf020.json
  scripts/
    start_mongo_local.ps1
    start_backend.ps1
    run_backend_8000.ps1
  SYSTEM_TEST_GUIDE.md
  README.md
```

## Cách Chạy Demo

### 1. Start MongoDB local

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_mongo_local.ps1
```

Mặc định MongoDB chạy tại:

```text
mongodb://127.0.0.1:27017
```

### 2. Start backend FastAPI

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1
```

Backend chạy tại và cũng serve luôn giao diện React đã build:

```text
http://127.0.0.1:8000
```

Mở demo bằng URL này:

```text
http://127.0.0.1:8000
```

Nên dùng `127.0.0.1` thay vì `localhost` trên Windows để tránh lỗi browser ưu tiên IPv6 làm kết nối bị từ chối.

### 3. Frontend React/Vite Cho Dev Tuỳ Chọn

```powershell
cd frontend
npm install
npm.cmd run dev
```

Vite dev server mặc định chạy tại:

```text
http://127.0.0.1:5173
```

Chế độ này chỉ cần khi sửa frontend. Demo ổn định nhất là mở `http://127.0.0.1:8000`.

Frontend dev gọi backend qua:

```text
VITE_API_BASE=http://127.0.0.1:8000
```

## API Chính

```text
GET  /health
POST /predict
GET  /predictions
GET  /predictions/{id}
DELETE /predictions/{id}
GET  /sample-image
```

## Train Lại YOLO Trên Kaggle GPU

Dữ liệu thô hiện có đủ DICOM và XML annotation để rebuild nhánh localization từ raw data. Pipeline YOLO dùng contour `edgeMap` trong XML để suy ra bounding box nốt phổi, giữ split theo `patient_id`, train trên GPU và xuất checkpoint/metrics.

Script Kaggle-friendly:

```text
scripts/kaggle_yolo_pipeline.py
```

Ví dụ chạy trong Kaggle Notebook sau khi add raw LIDC/Kaggle dataset vào `/kaggle/input/...`:

```bash
pip install -r requirements-yolo.txt

python scripts/kaggle_yolo_pipeline.py \
  --raw-root "/kaggle/input/<ten-dataset-raw>/archive (1)" \
  --output-dir "/kaggle/working/yolo_nodule" \
  --run-project "/kaggle/working/yolo_runs" \
  --run-name "yolo_nodule_gpu_s640_e80" \
  --model "yolov8s.pt" \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --conf 0.20 \
  --export-zip
```

Artifact chính sau khi chạy:

```text
/kaggle/working/yolo_runs/<run-name>/weights/best.pt
/kaggle/working/yolo_runs/<run-name>/kaggle_test_metrics.json
/kaggle/working/yolo_runs/<run-name>/detection_details_conf020.json
/kaggle/working/yolo_runs/<run-name>_test_predictions/
/kaggle/working/yolo_runs/<run-name>_artifacts.zip
```

Chỉ nên promote YOLO mới nếu test split patient-level tốt hơn production hiện tại, đặc biệt recall tăng mà false positive không tăng mất kiểm soát.

## Kết Quả Production Chính

Classification production trên test split patient-level:

```text
Accuracy: 0.8400
Macro F1: 0.8395
high_malignancy F1: 0.8235
low_malignancy F1: 0.8571
medium_risk F1: 0.8378
```

YOLO production mới tại `imgsz=640`, `conf=0.12`:

```text
Box precision@IoU0.5: 0.7037
Box recall@IoU0.5: 0.5528
Matched GT boxes: 748 / 1353
Missed GT boxes: 605
False positive boxes: 315
Avg predicted boxes/image: 0.9235
```

## Lưu Ý Học Thuật

- Đây là hệ thống demo nghiên cứu, không phải thiết bị y tế.
- Kết quả chỉ nên xem như tham khảo học thuật.
- Lớp `medium_risk` là vùng trung gian, cần diễn giải thận trọng hơn hai cực low/high.
- YOLO localization hỗ trợ trực quan hóa vùng nghi ngờ, không phải bằng chứng chẩn đoán độc lập.
- Nếu classification dự đoán high nhưng YOLO không phát hiện box, hệ thống hiển thị cảnh báo bất nhất để người dùng không diễn giải quá mức.
