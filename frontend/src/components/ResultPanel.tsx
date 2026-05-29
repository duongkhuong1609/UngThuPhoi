import type { FormState, PredictionResponse } from '../types';
import { ImageWithFallback } from './ImageWithFallback';
import { buildPredictionNarrative } from '../data/predictionNarrative';
import { printElementById } from '../utils/printReport';

type ImageInfo = {
  previewUrl: string | null;
  source: 'sample' | 'upload' | 'none';
  mimeType: string | null;
  width: number | null;
  height: number | null;
  isValid: boolean;
};

type Props = {
  loading: boolean;
  error: string | null;
  result: PredictionResponse | null;
  form: FormState;
  imageInfo: ImageInfo;
};

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

export function ResultPanel({ loading, error, result, form, imageInfo }: Props) {
  const narrative = result
    ? buildPredictionNarrative({
        riskLevel: result.risk_level,
        probabilities: result.probabilities,
        age: result.input_summary.age,
        smokingStatus: result.input_summary.smoking_status,
        familyHistory: result.input_summary.family_history,
        symptomScore: result.input_summary.symptom_score,
        tumorSize: result.input_summary.tumor_size_effective ?? result.input_summary.tumor_size,
        tumorSizeMissing: result.input_summary.tumor_size_missing,
        tumorSizeImputed: result.input_summary.tumor_size_imputed,
        localizationAvailable: result.localization?.available,
        localizationBoxCount: result.localization?.box_count,
        noBoxHighRisk: result.localization_classification_consistency?.no_box_high_risk,
      })
    : null;

  const displayAge = result?.input_summary?.age ?? form.age;
  const displaySex = result?.input_summary?.sex ?? form.sex;
  const displaySmoking = result?.input_summary?.smoking_status ?? form.smoking_status;
  const displayFamilyHistory = result?.input_summary?.family_history ?? form.family_history;
  const displaySymptomScore = result?.input_summary?.symptom_score ?? form.symptom_score;
  const imageType =
    imageInfo.mimeType ?? (imageInfo.previewUrl?.toLowerCase().endsWith('.png') ? 'image/png' : 'Không xác định');
  const imageSize = imageInfo.width && imageInfo.height ? `${imageInfo.width} x ${imageInfo.height}` : 'Chưa xác định';
  const displayTumorSize = result?.input_summary?.tumor_size ?? form.tumor_size;
  const tumorSizeText = result?.input_summary?.tumor_size_missing
    ? 'Chưa nhập'
    : result?.input_summary?.tumor_size_imputed
      ? 'Giá trị chưa hợp lệ, hệ thống đã dùng giá trị thay thế an toàn'
      : displayTumorSize
        ? `${displayTumorSize} mm`
        : 'Chưa nhập';
  const yoloSummary = result?.localization
    ? `Khoanh vùng nghi ngờ: phát hiện ${result.localization.box_count} vùng nghi ngờ.`
    : 'Khoanh vùng nghi ngờ sẽ hiển thị sau khi dự đoán.';

  return (
    <section id={result ? 'prediction-print-report' : undefined} className="panel panel--result panel--result-merged">
      <div className="panel__header">
        <div>
          <h2>Phân tích kết quả</h2>
        </div>
        {result && (
          <button
            className="button button--small print-hidden"
            type="button"
            onClick={() => printElementById('prediction-print-report', `Báo cáo dự đoán - ${form.patient_name || 'bệnh nhân'}`)}
          >
            In PDF
          </button>
        )}
      </div>

      {error && <div className="notice notice--error">{error}</div>}

      <div className="result-summary-grid">
        <div className="status-card">
          <span className={`status-dot ${loading ? 'status-dot--loading' : result ? 'status-dot--done' : 'status-dot--idle'}`} />
          <div>
            <p className="status-card__label">Trạng thái xử lý</p>
            <p className="status-card__value">{loading ? 'Đang phân tích...' : result ? 'Hoàn tất' : 'Chờ dữ liệu'}</p>
          </div>
        </div>

        {result && narrative ? (
          <div
            className={`risk-banner ${
              result.risk_level === 'high_malignancy'
                ? 'risk-banner--high'
                : result.risk_level === 'medium_risk'
                  ? 'risk-banner--medium'
                  : 'risk-banner--low'
            }`}
          >
            <p className="risk-banner__label">Kết luận hiện tại</p>
            <h3>{narrative.heading}</h3>
            <p>{narrative.explanation}</p>
            {narrative.supportingNote && <p className="risk-banner__note">{narrative.supportingNote}</p>}
          </div>
        ) : (
          <div className="empty-state result-summary-empty">Kết quả sẽ hiển thị sau khi bạn nhấn Dự đoán.</div>
        )}
      </div>

      {!loading && !result && !error && <div className="empty-state">Chưa có dữ liệu để phân tích.</div>}

      {result && narrative && (
        <div className="result-detail-grid">
          <div className="analysis-card">
            <h3>Ảnh và vùng nghi ngờ</h3>
            <p className="image-compare-label">{yoloSummary}</p>
            {result.localization?.annotated_image ? (
              <ImageWithFallback
                className="analysis-image analysis-image--localized"
                src={result.localization.annotated_image}
                alt="Ảnh CT có vùng nghi ngờ"
                fallbackText="Không hiển thị được ảnh YOLO overlay."
                logContext={{ source: result.localization.source, box_count: result.localization.box_count }}
              />
            ) : (
              <div className="analysis-image analysis-image--empty">
                {result.localization?.available === false
                  ? 'YOLO localization chưa khả dụng.'
                  : 'Vùng khoanh sẽ hiển thị sau khi dự đoán.'}
              </div>
            )}

            <div className="analysis-meta-grid">
              <div><span>Loại ảnh</span><strong>{imageType}</strong></div>
              <div><span>Kích thước</span><strong>{imageSize}</strong></div>
              <div>
                <span>Trạng thái ảnh</span>
                <strong className={imageInfo.isValid ? 'text-ok' : 'text-warn'}>
                  {imageInfo.isValid ? 'Hợp lệ để dự đoán' : 'Cần chọn ảnh hợp lệ'}
                </strong>
              </div>
            </div>

            {result.localization?.available && result.localization.box_count === 0 && (
              <p className="localization-note">
                Hệ thống chưa phát hiện vùng nghi ngờ vượt ngưỡng hiện tại. Hệ thống vẫn tiếp tục đọc ảnh và thông tin bệnh nhân để đưa ra kết quả.
              </p>
            )}
            {result.localization_classification_consistency?.no_box_high_risk && (
              <p className="localization-note localization-note--critical">
                {result.localization_classification_consistency.message}
              </p>
            )}
            {result.localization?.error && <p className="localization-error">{result.localization.error}</p>}
          </div>

          <div className="analysis-card">
            <h3>Thông tin bệnh nhân</h3>
            <div className="patient-groups">
              <div className="patient-group">
                <p className="patient-group__title">Thông tin cơ bản</p>
                <div className="patient-item"><span>Tuổi</span><strong>{displayAge}</strong></div>
                <div className="patient-item"><span>Giới tính</span><strong>{readableSex(displaySex)}</strong></div>
              </div>
              <div className="patient-group">
                <p className="patient-group__title">Yếu tố liên quan</p>
                <div className="patient-item"><span>Hút thuốc</span><strong>{readableSmoking(displaySmoking)}</strong></div>
                <div className="patient-item"><span>Tiền sử gia đình</span><strong>{readableFamily(displayFamilyHistory)}</strong></div>
                <div className="patient-item"><span>Kích thước nốt ác tính</span><strong>{tumorSizeText}</strong></div>
              </div>
              <div className="patient-group">
                <p className="patient-group__title">Triệu chứng</p>
                <div className="patient-item"><span>Điểm triệu chứng</span><strong>{displaySymptomScore}/10</strong></div>
              </div>
            </div>
          </div>

          <div className="analysis-card result-card-span">
            <h3>Lời khuyên cho bệnh nhân</h3>
            <ul className="advice-list">
              {narrative.advice.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>

          <div className="chart-card result-card-span">
            <h3>Phân bố kết quả</h3>
            {probabilityBar(
              'Xác suất ác tính cao',
              result.probabilities.high_malignancy,
              'high',
              'Tỷ lệ này càng cao thì mô hình càng nghiêng về nhóm nguy cơ cao.',
            )}
            {probabilityBar(
              'Xác suất trung gian',
              result.probabilities.medium_risk,
              'medium',
              'Tỷ lệ này thường tăng khi mẫu nằm gần ranh giới giữa các nhóm.',
            )}
            {probabilityBar(
              'Xác suất ác tính thấp',
              result.probabilities.low_malignancy,
              'low',
              'Tỷ lệ này càng cao thì mô hình càng nghiêng về nhóm nguy cơ thấp.',
            )}
          </div>
        </div>
      )}
    </section>
  );
}
