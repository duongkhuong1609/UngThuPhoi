import type { FormState, PredictionResponse } from '../types';
import { ImageWithFallback } from './ImageWithFallback';

type ImageInfo = {
  previewUrl: string | null;
  source: 'sample' | 'upload' | 'none';
  mimeType: string | null;
  width: number | null;
  height: number | null;
  isValid: boolean;
};

type Props = {
  result: PredictionResponse | null;
  form: FormState;
  imageInfo: ImageInfo;
};

function probabilityBar(label: string, value: number, tone: 'low' | 'medium' | 'high', hint: string) {
  return (
    <div className="prob-row" key={label}>
      <div className="prob-row__top">
        <span>{label}</span>
        <strong>{(value * 100).toFixed(2)}%</strong>
      </div>
      <div className="prob-track">
        <div className={`prob-fill prob-fill--${tone}`} style={{ width: `${Math.max(4, value * 100)}%` }} />
      </div>
      <p className="prob-row__hint">{hint}</p>
    </div>
  );
}

function readableSex(value: string) {
  return value === 'male' ? 'Nam' : value === 'female' ? 'Nữ' : value;
}

function readableSmoking(value: string) {
  if (value === 'never') return 'Không hút';
  if (value === 'former') return 'Đã từng hút';
  if (value === 'current') return 'Đang hút';
  return value;
}

function readableFamily(value: string) {
  if (value === 'yes') return 'Có';
  if (value === 'no') return 'Không';
  return value;
}

export function AnalyticsPanel({ result, form, imageInfo }: Props) {
  const sourceLabel = imageInfo.source === 'sample' ? 'Ảnh mẫu' : imageInfo.source === 'upload' ? 'Ảnh tải lên' : 'Chưa có ảnh';
  const imageType = imageInfo.mimeType ?? (imageInfo.previewUrl?.toLowerCase().endsWith('.png') ? 'image/png' : 'Không xác định');
  const imageSize = imageInfo.width && imageInfo.height ? `${imageInfo.width} x ${imageInfo.height}` : 'Chưa xác định';
  const tumorSizeText = result?.input_summary?.tumor_size_missing
    ? 'Chưa nhập'
    : result?.input_summary?.tumor_size_imputed
      ? 'Giá trị chưa hợp lệ, hệ thống đã dùng cách thay thế an toàn'
      : form.tumor_size
        ? `${form.tumor_size} mm`
        : 'Chưa nhập';
  const yoloStatus = result?.localization
    ? result.localization.available
      ? result.localization.box_count > 0
        ? `Phát hiện ${result.localization.box_count} vùng nghi ngờ`
        : 'Không phát hiện vùng nghi ngờ rõ ràng'
      : 'YOLO chưa khả dụng'
    : 'Chưa chạy';

  return (
    <section className="panel panel--analytics">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Cột 3</p>
          <h2>Phân tích dữ liệu</h2>
        </div>
        <p className="panel__subtitle">Phân tích ảnh, thông tin bệnh nhân và các tín hiệu chính từ mô hình.</p>
      </div>

      <div className="analysis-card">
        <h3>Phân tích dữ liệu ảnh</h3>
        <div className="image-compare-grid">
          <div>
            <p className="image-compare-label">Ảnh CT đầu vào</p>
            {imageInfo.previewUrl ? (
              <ImageWithFallback
                className="analysis-image"
                src={imageInfo.previewUrl}
                alt="Ảnh CT đầu vào"
                fallbackText="Không hiển thị được ảnh CT đầu vào."
                logContext={{ source: imageInfo.source }}
              />
            ) : (
              <div className="analysis-image analysis-image--empty">Chưa có ảnh đầu vào.</div>
            )}
          </div>
          <div>
            <p className="image-compare-label">YOLO khoanh vùng nghi ngờ</p>
            {result?.localization?.annotated_image ? (
              <ImageWithFallback
                className="analysis-image analysis-image--localized"
                src={result.localization.annotated_image}
                alt="Ảnh CT có vùng nghi ngờ"
                fallbackText="Không hiển thị được ảnh YOLO overlay."
                logContext={{ source: result.localization.source, box_count: result.localization.box_count }}
              />
            ) : (
              <div className="analysis-image analysis-image--empty">
                {result?.localization?.available === false
                  ? 'YOLO localization chưa khả dụng.'
                  : 'Vùng khoanh sẽ hiển thị sau khi dự đoán.'}
              </div>
            )}
          </div>
        </div>
        <div className="analysis-meta-grid">
          <div><span>Loại ảnh</span><strong>{imageType}</strong></div>
          <div><span>Kích thước</span><strong>{imageSize}</strong></div>
          <div><span>Nguồn ảnh</span><strong>{sourceLabel}</strong></div>
          <div>
            <span>Trạng thái</span>
            <strong className={imageInfo.isValid ? 'text-ok' : 'text-warn'}>{imageInfo.isValid ? 'Hợp lệ để dự đoán' : 'Cần chọn ảnh hợp lệ'}</strong>
          </div>
        </div>
        {result?.localization && (
          <div className="localization-summary">
            <div>
              <span>Số vùng nghi ngờ</span>
              <strong>{result.localization.box_count}</strong>
            </div>
            <div>
              <span>Checkpoint YOLO</span>
              <strong>{result.localization.checkpoint ?? 'Không xác định'}</strong>
            </div>
          </div>
        )}
        <div className="pipeline-steps">
          <div className={imageInfo.isValid ? 'pipeline-step pipeline-step--ok' : 'pipeline-step pipeline-step--warn'}>
            <span>1. Kiểm tra ảnh</span>
            <strong>{imageInfo.isValid ? 'Ảnh hợp lệ' : 'Chưa hợp lệ hoặc chưa chọn ảnh'}</strong>
          </div>
          <div className={result?.localization?.box_count ? 'pipeline-step pipeline-step--ok' : 'pipeline-step pipeline-step--warn'}>
            <span>2. YOLO localization</span>
            <strong>{yoloStatus}</strong>
          </div>
          <div className={result ? 'pipeline-step pipeline-step--ok' : 'pipeline-step pipeline-step--idle'}>
            <span>3. Phân loại đa phương thức</span>
            <strong>{result ? 'Đã hoàn tất' : 'Chờ dự đoán'}</strong>
          </div>
        </div>
        {result?.localization?.available && result.localization.box_count === 0 && (
          <p className="localization-note">
            YOLO chưa thấy vùng nghi ngờ vượt ngưỡng hiện tại. Hệ thống vẫn tiếp tục đọc ảnh và thông tin bệnh nhân để đưa ra kết quả.
          </p>
        )}
        {result?.localization_classification_consistency?.no_box_high_risk && (
          <p className="localization-note localization-note--critical">
            {result.localization_classification_consistency.message}
          </p>
        )}
        {result?.localization?.error && <p className="localization-error">{result.localization.error}</p>}
      </div>

      <div className="analysis-card">
        <h3>Phân tích dữ liệu bệnh nhân</h3>
        <div className="patient-groups">
          <div className="patient-group">
            <p className="patient-group__title">Thông tin cơ bản</p>
            <div className="patient-item"><span>Tuổi</span><strong>{form.age}</strong></div>
            <div className="patient-item"><span>Giới tính</span><strong>{readableSex(form.sex)}</strong></div>
          </div>
          <div className="patient-group">
            <p className="patient-group__title">Yếu tố liên quan</p>
            <div className="patient-item"><span>Hút thuốc</span><strong>{readableSmoking(form.smoking_status)}</strong></div>
            <div className="patient-item"><span>Tiền sử gia đình</span><strong>{readableFamily(form.family_history)}</strong></div>
            <div className="patient-item"><span>Kích thước nốt ác tính</span><strong>{tumorSizeText}</strong></div>
          </div>
          <div className="patient-group">
            <p className="patient-group__title">Triệu chứng</p>
            <div className="patient-item"><span>Điểm triệu chứng</span><strong>{form.symptom_score}/10</strong></div>
          </div>
        </div>
      </div>

      {result ? (
        <div className="chart-card">
          <h3>Phân bố kết quả</h3>
          {probabilityBar('Xác suất ác tính cao', result.probabilities.high_malignancy, 'high', 'Tỷ lệ này càng cao thì mô hình càng nghiêng về nhóm nguy cơ cao.')}
          {probabilityBar('Xác suất trung gian', result.probabilities.medium_risk, 'medium', 'Tỷ lệ này cao hơn khi mẫu nằm gần ranh giới giữa các nhóm.')}
          {probabilityBar('Xác suất ác tính thấp', result.probabilities.low_malignancy, 'low', 'Tỷ lệ này càng cao thì mô hình càng nghiêng về nhóm nguy cơ thấp.')}
        </div>
      ) : (
        <div className="empty-state">Các chỉ số sẽ hiển thị sau khi có kết quả dự đoán.</div>
      )}
    </section>
  );
}
