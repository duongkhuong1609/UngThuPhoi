import { useRef, useState } from 'react';
import type { FormState, ValidationErrors } from '../types';
import { ImageWithFallback } from './ImageWithFallback';
import SampleSelector from './SampleSelector';

type Props = {
  form: FormState;
  errors: ValidationErrors;
  onChange: (patch: Partial<FormState>) => void;
  onFileChange: (file: File | null) => void;
  onPredict: () => void;
  onReset: () => void;
  disabled: boolean;
  onSampleSelect?: (row: any) => void;
  sampleLoaded?: boolean;
  onRandomSample?: () => void;
  onRandomizeParams?: () => void;
};

const sexOptions = [
  { label: 'Nữ', value: 'female' },
  { label: 'Nam', value: 'male' },
] as const;

const smokingOptions = [
  { label: 'Không hút', value: 'never' },
  { label: 'Đã từng hút', value: 'former' },
  { label: 'Đang hút', value: 'current' },
] as const;

const familyOptions = [
  { label: 'Không có tiền sử', value: 'no' },
  { label: 'Có tiền sử', value: 'yes' },
] as const;

export function InputPanel({ form, errors, onChange, onFileChange, onPredict, onReset, disabled, onSampleSelect, sampleLoaded, onRandomSample, onRandomizeParams }: Props) {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [showSamples, setShowSamples] = useState(false);
  const [sampleDeckVersion, setSampleDeckVersion] = useState(0);

  const triggerFile = () => fileRef.current?.click();
  const showRandomSamples = () => {
    setSampleDeckVersion((version) => version + 1);
    setShowSamples(true);
  };

  const handleSampleSelect = (row: any) => {
    setShowSamples(false);
    if (onSampleSelect) onSampleSelect(row);
  };

  return (
    <section className="panel panel--input">
      <div className="panel__header">
        <div>
          <h2>Nhập dữ liệu</h2>
        </div>
        <p className="panel__subtitle">Chọn ảnh CT và nhập thông tin bệnh nhân để chạy dự đoán.</p>
      </div>

      <div className="input-actions">
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
          style={{ display: 'none' }}
        />
        <button className="button button--ghost" type="button" onClick={showRandomSamples}>
          Dùng ảnh mẫu
        </button>
        <button className="button button--ghost" type="button" onClick={triggerFile}>
          Tải ảnh lên
        </button>
        {sampleLoaded && (
          <>
            <button className="button button--small" type="button" onClick={onRandomSample}>
              Mẫu ngẫu nhiên
            </button>
            <button className="button button--small" type="button" onClick={onRandomizeParams}>
              Thông số ngẫu nhiên
            </button>
          </>
        )}
      </div>

      {showSamples && <SampleSelector key={sampleDeckVersion} onSelect={handleSampleSelect} />}

      <div className="preview-card">
        <div className="preview-card__header">
          <div>
            <p className="preview-card__eyebrow">Ảnh đầu vào</p>
            <h3>CT Preview</h3>
          </div>
          {form.imagePreview && (
            <span className="chip chip--muted">
              {form.sample_path ? 'Nguồn mẫu' : 'Nguồn upload'}
            </span>
          )}
        </div>
        {form.imagePreview ? (
          <ImageWithFallback
            className="preview-card__image"
            src={form.imagePreview}
            alt="CT preview"
            fallbackText="Không hiển thị được ảnh preview. Hãy nạp lại mẫu hoặc chọn ảnh khác."
            logContext={{ source: form.sample_path ? 'sample' : 'upload', sample_path: form.sample_path }}
          />
        ) : (
          <div className="preview-card__empty">Chưa có ảnh. Hãy chọn ảnh mẫu hoặc tải ảnh CT lên.</div>
        )}
        {form.image && <p className="preview-card__filename">Đã nạp: {form.image.name}</p>}
      </div>

      {errors.image && <p className="field__error">{errors.image}</p>}

      <div className="field">
        <label>Tên bệnh nhân</label>
        <input
          type="text"
          value={form.patient_name}
          onChange={(e) => onChange({ patient_name: e.target.value })}
          placeholder="Ví dụ: Nguyễn Văn A"
        />
        {errors.patient_name && <p className="field__error">{errors.patient_name}</p>}
      </div>

      <div className="grid grid--two">
        <div className="field">
          <label>Tuổi</label>
          <input
            type="number"
            value={form.age}
            onChange={(e) => onChange({ age: e.target.value })}
            min="20"
            max="90"
            placeholder="Nhập tuổi"
          />
          {errors.age && <p className="field__error">{errors.age}</p>}
        </div>

        <div className="field">
          <label>Giới tính</label>
          <select value={form.sex} onChange={(e) => onChange({ sex: e.target.value as FormState['sex'] })}>
            <option value="">Chọn giới tính</option>
            {sexOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          {errors.sex && <p className="field__error">{errors.sex}</p>}
        </div>
      </div>

      <div className="grid grid--two">
        <div className="field">
          <label>Tình trạng hút thuốc</label>
          <select
            value={form.smoking_status}
            onChange={(e) => onChange({ smoking_status: e.target.value as FormState['smoking_status'] })}
          >
            <option value="">Chọn tình trạng hút thuốc</option>
            {smokingOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          {errors.smoking_status && <p className="field__error">{errors.smoking_status}</p>}
        </div>

        <div className="field">
          <label>Kích thước nốt ác tính (mm, tùy chọn)</label>
          <input
            type="number"
            step="0.1"
            value={form.tumor_size}
            onChange={(e) => onChange({ tumor_size: e.target.value })}
            min="2"
            max="40"
            placeholder="Có thể bỏ trống nếu chưa có thông tin"
          />
          {errors.tumor_size && <p className="field__error">{errors.tumor_size}</p>}
        </div>
      </div>

      <div className="grid grid--two">
        <div className="field">
          <label className="label-with-hint">
            <span>Tiền sử gia đình</span>
            <span
              className="label-hint"
              title="Ý nghĩa: có/không có tiền sử gia đình liên quan ung thư phổi. Không phải tiền sử hút thuốc."
              aria-label="Giải thích Tiền sử gia đình"
            >
              ?
            </span>
          </label>
          <select
            value={form.family_history}
            onChange={(e) => onChange({ family_history: e.target.value as FormState['family_history'] })}
          >
            <option value="">Chọn tiền sử gia đình</option>
            {familyOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          {errors.family_history && <p className="field__error">{errors.family_history}</p>}
        </div>

        <div className="field">
          <label className="label-with-hint">
            <span>Mức độ triệu chứng</span>
            <span
              className="label-hint"
              title="Thang điểm 1-10 phản ánh mức độ biểu hiện triệu chứng: 1 là rất nhẹ, 10 là rất nặng."
              aria-label="Giải thích Mức độ triệu chứng"
            >
              ?
            </span>
          </label>
          <input
            type="number"
            value={form.symptom_score}
            onChange={(e) => onChange({ symptom_score: e.target.value })}
            min="1"
            max="10"
            step="1"
            placeholder="1-10"
          />
          {errors.symptom_score && <p className="field__error">{errors.symptom_score}</p>}
        </div>
      </div>

      <div className="action-row">
        <button className="button button--primary" type="button" onClick={onPredict} disabled={disabled}>
          Dự đoán
        </button>
        <button className="button button--ghost" type="button" onClick={onReset} disabled={disabled}>
          Đặt lại
        </button>
      </div>
    </section>
  );
}


