import { Fragment, useMemo, useState } from 'react';
import type { PredictionHistoryRecord } from '../types';
import { sampleImageUrl } from '../api';
import { ImageWithFallback } from './ImageWithFallback';
import { buildPredictionNarrative, labelRisk } from '../data/predictionNarrative';
import { printElementById } from '../utils/printReport';

type Props = {
  items: PredictionHistoryRecord[];
  selected: PredictionHistoryRecord | null;
  loading: boolean;
  error: string | null;
  onSelect: (id: string) => void;
  onCollapse: () => void;
  onDelete: (id: string) => void;
  onRefresh: () => void;
};

type RiskFilter = 'all' | 'high_malignancy' | 'medium_risk' | 'low_malignancy';

function readableSex(value: string): string {
  if (value === 'male') return 'Nam';
  if (value === 'female') return 'Nữ';
  return value;
}

function readableSmoking(value: string): string {
  if (value === 'never') return 'Không hút';
  if (value === 'former') return 'Đã từng hút';
  if (value === 'current') return 'Đang hút';
  return value;
}

function readableFamily(value: string): string {
  if (value === 'yes') return 'Có';
  if (value === 'no') return 'Không';
  return value;
}

function readableSource(value: string): string {
  if (value === 'sample') return 'Ảnh mẫu';
  if (value === 'upload') return 'Ảnh tải lên';
  return value;
}

function formatTime(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString('vi-VN');
}

function formatDateKey(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return '';
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function formatMonthKey(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return '';
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  return `${y}-${m}`;
}

function formatTumorSize(item: PredictionHistoryRecord): string {
  if (item.tumor_size_missing) return 'Chưa nhập';
  if (item.tumor_size) return `${item.tumor_size} mm`;
  return 'Chưa nhập';
}

function maxProbability(item: PredictionHistoryRecord): string {
  return `${(
    Math.max(
      item.probabilities.high_malignancy,
      item.probabilities.medium_risk,
      item.probabilities.low_malignancy,
    ) * 100
  ).toFixed(2)}%`;
}

function predictedProbability(item: PredictionHistoryRecord): number {
  if (item.predicted_label === 'high_malignancy') return item.probabilities.high_malignancy;
  if (item.predicted_label === 'medium_risk') return item.probabilities.medium_risk;
  if (item.predicted_label === 'low_malignancy') return item.probabilities.low_malignancy;
  return Math.max(item.probabilities.high_malignancy, item.probabilities.medium_risk, item.probabilities.low_malignancy);
}

function historyImageSrc(item: PredictionHistoryRecord): string | null {
  const yoloOverlay = item.localization?.result?.annotated_image;
  if (yoloOverlay) return yoloOverlay;
  const path = item.localization?.sample_path || item.sample_path || item.image_path;
  if (!path || path === 'unknown_image') return null;
  if (path.startsWith('data/')) return sampleImageUrl(path, 'history-inline-detail');
  return null;
}

function historyImageNote(item: PredictionHistoryRecord): string {
  if (item.source === 'upload' && !historyImageSrc(item)) {
    return 'Bản ghi upload hiện chỉ lưu tên file, chưa lưu lại ảnh gốc để xem lại.';
  }
  if (item.localization?.result?.annotated_image) {
    return 'Đây là ảnh đã được hệ thống khoanh vùng nghi ngờ ở lần dự đoán này.';
  }
  return 'Bản ghi này chưa có ảnh khoanh vùng, hệ thống đang hiển thị ảnh CT gốc.';
}

function historyPrintId(id: string): string {
  return `history-print-${id.replace(/[^a-zA-Z0-9_-]/g, '')}`;
}

function HistoryInlineDetail({ item }: { item: PredictionHistoryRecord }) {
  const imageSrc = historyImageSrc(item);
  const reportId = historyPrintId(item.id);
  const resultPercent = (predictedProbability(item) * 100).toFixed(2);
  const narrative = buildPredictionNarrative({
    riskLevel: item.predicted_label,
    probabilities: item.probabilities,
    age: item.age,
    smokingStatus: item.smoking_status,
    familyHistory: item.family_history,
    symptomScore: item.symptom_score,
    tumorSize: item.tumor_size_effective ?? item.tumor_size,
    tumorSizeMissing: item.tumor_size_missing,
    tumorSizeImputed: item.tumor_size_imputed,
    localizationAvailable: item.localization?.result?.available,
    localizationBoxCount: item.localization?.result?.box_count,
    noBoxHighRisk: item.predicted_label === 'high_malignancy' && (item.localization?.result?.box_count ?? 0) === 0,
  });

  return (
    <div id={reportId} className="history-detail history-detail--inline">
      <div className="history-detail-actions print-hidden">
        <button
          type="button"
          className="button button--small"
          onClick={() => printElementById(reportId, `Báo cáo lịch sử - ${item.patient_name || item.id}`)}
        >
          In PDF
        </button>
      </div>
      <div className="history-detail-grid history-detail-grid--compact">
        <div><span>Bệnh nhân</span><strong>{item.patient_name}</strong></div>
        <div><span>Thời gian</span><strong>{formatTime(item.timestamp)}</strong></div>
        <div><span>Kết quả dự đoán</span><strong>{labelRisk(item.predicted_label)} - {resultPercent}%</strong></div>
        <div><span>Tuổi</span><strong>{item.age}</strong></div>
        <div><span>Giới tính</span><strong>{readableSex(item.sex)}</strong></div>
        <div><span>Nguồn ảnh</span><strong>{readableSource(item.source)}</strong></div>
        <div><span>Tình trạng hút thuốc</span><strong>{readableSmoking(item.smoking_status)}</strong></div>
        <div><span>Tiền sử gia đình</span><strong>{readableFamily(item.family_history)}</strong></div>
        <div><span>Kích thước nốt ác tính</span><strong>{formatTumorSize(item)}</strong></div>
        <div><span>Điểm triệu chứng</span><strong>{item.symptom_score}/10</strong></div>
      </div>

      <div className={`history-insight-card ${item.predicted_label === 'medium_risk' ? 'history-insight-card--medium' : ''}`}>
        <h4>{narrative.heading}</h4>
        <p>{narrative.explanation}</p>
        {narrative.supportingNote && <p className="history-insight-card__note">{narrative.supportingNote}</p>}
        <div className="history-advice-block">
          <h5>Lời khuyên cho bệnh nhân</h5>
          <ul className="advice-list advice-list--history">
            {narrative.advice.map((advice) => (
              <li key={advice}>{advice}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="history-image-detail">
        <div className="history-image-copy">
          <h4>Vùng nghi ngờ</h4>
          <p>{historyImageNote(item)}</p>
          {item.localization?.result?.box_count !== undefined && (
            <p>Phát hiện: {item.localization.result.box_count} vùng nghi ngờ.</p>
          )}
        </div>
        {imageSrc ? (
          <ImageWithFallback
            className="history-detail-image"
            src={imageSrc}
            alt="Ảnh CT đã dùng trong lịch sử dự đoán"
            fallbackText="Không hiển thị được ảnh CT trong lịch sử."
            logContext={{ history_id: item.id, image_path: item.image_path }}
          />
        ) : (
          <div className="history-image-placeholder">
            Không có ảnh lưu trữ để xem lại cho bản ghi này.
          </div>
        )}
      </div>
    </div>
  );
}

export function HistoryPanel({ items, selected, loading, error, onSelect, onCollapse, onDelete, onRefresh }: Props) {
  const [nameQuery, setNameQuery] = useState('');
  const [dateFilter, setDateFilter] = useState('');
  const [monthFilter, setMonthFilter] = useState('');
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all');

  const filteredItems = useMemo(() => {
    const normalizedName = nameQuery.trim().toLowerCase();

    return items.filter((item) => {
      const matchesName =
        !normalizedName ||
        item.patient_name.toLowerCase().includes(normalizedName) ||
        item.patient_id.toLowerCase().includes(normalizedName);
      const matchesRisk = riskFilter === 'all' || item.predicted_label === riskFilter;
      const matchesDate = !dateFilter || formatDateKey(item.timestamp) === dateFilter;
      const matchesMonth = !monthFilter || formatMonthKey(item.timestamp) === monthFilter;
      return matchesName && matchesRisk && matchesDate && matchesMonth;
    });
  }, [items, nameQuery, dateFilter, monthFilter, riskFilter]);

  const clearFilters = () => {
    setNameQuery('');
    setDateFilter('');
    setMonthFilter('');
    setRiskFilter('all');
  };

  const hasFilters = Boolean(nameQuery || dateFilter || monthFilter || riskFilter !== 'all');

  return (
    <section className="panel history-panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Lịch sử</p>
          <h2>Lịch sử dự đoán</h2>
        </div>
        <p className="panel__subtitle">Tìm theo tên bệnh nhân, ngày hoặc tháng dự đoán, và lọc nhanh theo mức nguy cơ.</p>
      </div>

      <div className="history-filter-card">
        <div className="history-filter-grid">
          <div className="field history-filter-field">
            <label>Tìm tên bệnh nhân</label>
            <input
              type="search"
              value={nameQuery}
              onChange={(e) => setNameQuery(e.target.value)}
              placeholder="Nhập tên hoặc mã bệnh nhân"
            />
          </div>

          <div className="field history-filter-field">
            <label>Ngày dự đoán</label>
            <input
              type="date"
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value)}
            />
          </div>

          <div className="field history-filter-field">
            <label>Tháng dự đoán</label>
            <input
              type="month"
              value={monthFilter}
              onChange={(e) => setMonthFilter(e.target.value)}
            />
          </div>

          <div className="field history-filter-field">
            <label>Loại nguy cơ</label>
            <select
              value={riskFilter}
              onChange={(e) => setRiskFilter(e.target.value as RiskFilter)}
            >
              <option value="all">Tất cả</option>
              <option value="high_malignancy">Nguy cơ cao</option>
              <option value="medium_risk">Nguy cơ trung gian</option>
              <option value="low_malignancy">Nguy cơ thấp</option>
            </select>
          </div>
        </div>

        <div className="history-filter-actions">
          <div className="history-filter-summary">
            Hiển thị {filteredItems.length}/{items.length} bản ghi
          </div>
          <button type="button" className="button button--small" onClick={onRefresh} disabled={loading}>
            Làm mới
          </button>
          <button type="button" className="button button--small" onClick={clearFilters} disabled={!hasFilters}>
            Xóa lọc
          </button>
        </div>
      </div>

      {error && <div className="notice notice--error">{error}</div>}

      {items.length === 0 ? (
        <div className="empty-state">Chưa có lịch sử dự đoán.</div>
      ) : filteredItems.length === 0 ? (
        <div className="empty-state">Không có bản ghi nào khớp với bộ lọc hiện tại.</div>
      ) : (
        <div className="history-table-wrap">
          <table className="history-table">
            <thead>
              <tr>
                <th>Bệnh nhân</th>
                <th>Thời gian</th>
                <th>Kết quả</th>
                <th>Xác suất</th>
                <th>Thông tin cơ bản</th>
                <th>Thao tác</th>
              </tr>
            </thead>
            <tbody>
              {filteredItems.map((item) => {
                const isOpen = selected?.id === item.id;
                const detailItem = isOpen ? selected : null;

                return (
                  <Fragment key={item.id}>
                    <tr className={isOpen ? 'history-row--open' : undefined}>
                      <td>
                        <strong>{item.patient_name}</strong>
                      </td>
                      <td>{formatTime(item.timestamp)}</td>
                      <td>{labelRisk(item.predicted_label)}</td>
                      <td>{maxProbability(item)}</td>
                      <td>
                        {item.age} tuổi, {readableSex(item.sex)}, {readableSmoking(item.smoking_status)}
                        <div className="history-sub">
                          Kích thước nốt ác tính: {formatTumorSize(item)}, tiền sử gia đình: {readableFamily(item.family_history)}, nguồn: {readableSource(item.source)}
                        </div>
                      </td>
                      <td>
                        <div className="history-row-actions">
                          {isOpen ? (
                            <button type="button" className="button button--small" onClick={onCollapse}>
                              Thu gọn
                            </button>
                          ) : (
                            <button type="button" className="button button--small" onClick={() => onSelect(item.id)}>
                              Xem
                            </button>
                          )}
                          <button type="button" className="button button--small" onClick={() => onDelete(item.id)}>
                            Xóa
                          </button>
                        </div>
                      </td>
                    </tr>
                    {detailItem && (
                      <tr className="history-detail-row">
                        <td colSpan={6}>
                          <HistoryInlineDetail item={detailItem} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
