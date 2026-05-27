# System Test Guide

File này dùng để kiểm tra nhanh hệ thống production sau khi dọn project.

## 1. Start MongoDB Local

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_mongo_local.ps1
```

Kỳ vọng:

```text
MongoDB chạy tại mongodb://127.0.0.1:27017
```

## 2. Start Backend

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1
```

Backend chạy tại:

```text
http://127.0.0.1:8000
```

Backend cũng serve luôn giao diện React đã build tại:

```text
http://127.0.0.1:8000
```

Nên dùng `127.0.0.1` thay vì `localhost` trên Windows để tránh lỗi browser ưu tiên IPv6.

## 3. Start Frontend Dev Tuỳ Chọn

```powershell
cd frontend
npm install
npm.cmd run dev
```

Vite dev server chạy tại:

```text
http://127.0.0.1:5173
```

Chỉ cần bước này khi phát triển giao diện. Khi demo, có thể bỏ qua và mở trực tiếp `http://127.0.0.1:8000`.

## 4. Test Health

Mở trình duyệt hoặc PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Checklist pass:

```text
model_loaded = true
checkpoint = multimodal_model_3class_distilled_student.pth
dataset_csv = dataset\multimodal_image_dataset_3class_roi.csv
yolo_localization_loaded = true
yolo_checkpoint = models\localization\yolo_nodule_kaggle_m640_e80_fixray2_best.pt
yolo_conf_threshold = 0.12
mongo_connected = true
startup_error = null
yolo_startup_error = null
```

Nếu `mongo_connected = false`, kiểm tra MongoDB local trước. Predict vẫn có thể chạy, nhưng lịch sử dự đoán sẽ không lưu được.

## 5. Test Predict Trên Giao Diện

1. Mở giao diện tại `http://127.0.0.1:8000`.
2. Bấm dùng ảnh mẫu hoặc upload ảnh CT phổi hợp lệ.
3. Nhập thông tin bệnh nhân bắt buộc.
4. `tumor_size` có thể nhập hoặc để trống.
5. Bấm Predict.

Checklist pass:

```text
Ảnh preview hiển thị đúng
YOLO localization có trạng thái rõ ràng
Nếu có box, ảnh overlay bounding box hiển thị
Classification trả nhãn low/medium/high
Xác suất từng lớp hiển thị hợp lý
Không có lỗi Failed to fetch
```

## 6. Test YOLO Localization

Sau Predict, kiểm tra khu vực localization:

```text
box_count hiển thị số box phát hiện
annotated_image hiển thị nếu backend trả overlay
Nếu box_count = 0, giao diện hiển thị thông báo không phát hiện vùng nghi ngờ rõ ràng
```

Nếu classification là high nhưng YOLO không có box, hệ thống phải hiển thị cảnh báo bất nhất.

## 7. Test MongoDB History

Sau một lần Predict thành công:

1. Mở bảng lịch sử dự đoán trên frontend.
2. Kiểm tra bản ghi mới xuất hiện.
3. Bấm xem chi tiết.
4. Xóa bản ghi test nếu cần.

API tương ứng:

```text
GET /predictions
GET /predictions/{id}
DELETE /predictions/{id}
```

Checklist pass:

```text
Danh sách history tải được
Chi tiết history tải được
Xóa history hoạt động
Không báo Failed to fetch
```

## Checklist Cuối

```text
[ ] MongoDB local chạy
[ ] Backend /health pass
[ ] Frontend gọi đúng http://127.0.0.1:8000
[ ] Ảnh mẫu preview được
[ ] Upload ảnh CT phổi preview được
[ ] Predict trả kết quả classification
[ ] YOLO localization trả trạng thái/box/overlay
[ ] History MongoDB lưu được
[ ] Xem chi tiết history được
[ ] Xóa history được
```
