import { useEffect, useState } from 'react';
import { ImageWithFallback } from './ImageWithFallback';

type SampleRow = {
  image_path: string;
  preview_image_path?: string;
  patient_id: string;
  label: string;
  age: string;
  sex: string;
  smoking_status: string;
  tumor_size: string;
  family_history: string;
  symptom_score: string;
};

type Props = {
  onSelect: (row: SampleRow) => void;
  sampleCount?: number;
};

function dequote(value: string): string {
  const v = value.trim();
  if (v.length >= 2 && v.startsWith('"') && v.endsWith('"')) {
    return v.slice(1, -1);
  }
  return v;
}

function parseCsv(text: string): SampleRow[] {
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift()!.split(',').map(dequote);
  return lines.map((line) => {
    const parts = line.split(',').map(dequote);
    const obj: Record<string, string> = {};
    header.forEach((h, i) => (obj[h] = parts[i] ?? ''));
    return obj as SampleRow;
  });
}

function shuffleRows(rows: SampleRow[]): SampleRow[] {
  return [...rows]
    .map((row) => ({ row, key: Math.random() }))
    .sort((a, b) => a.key - b.key)
    .map((item) => item.row);
}

function SampleThumbnail({ row }: { row: SampleRow }) {
  const previewPath = (row.preview_image_path ?? '').trim() || row.image_path.trim();
  const src = `/sample-previews/${encodeURIComponent(row.patient_id)}.png`;

  return (
    <ImageWithFallback
      className="sample-thumb"
      src={src}
      alt={`Thumbnail CT ${row.patient_id}`}
      fallbackText="Không tải được thumbnail CT"
      logContext={{ patient_id: row.patient_id, label: row.label, previewPath, publicSrc: src }}
    />
  );
}

export function SampleSelector({ onSelect, sampleCount = 10 }: Props) {
  const [rows, setRows] = useState<SampleRow[] | null>(null);

  useEffect(() => {

    const url = new URL('../data/demo_samples_3class.csv', import.meta.url).toString();
    fetch(url)
      .then((r) => r.text())
      .then((txt) => setRows(shuffleRows(parseCsv(txt)).slice(0, sampleCount)))
      .catch((error) => {
        console.error('load sample csv failed', error);
        setRows([]);
      });
  }, [sampleCount]);

  if (!rows) return <div className="sample-selector">Đang tải danh sách mẫu...</div>;

  const renderCard = (row: SampleRow) => (
    <div key={row.patient_id + row.image_path} className="sample-card">
      <SampleThumbnail row={row} />
      <div className="sample-card__meta">
        <strong>{row.patient_id}</strong>
        <span className="chip chip--muted">Ảnh mẫu</span>
      </div>
      <div className="sample-card__body">
        <div>Tuổi: {row.age} • Giới tính: {row.sex}</div>
        <div>Hút thuốc: {row.smoking_status}</div>
        <div>Kích thước nốt ác tính: nhập tùy chọn khi dự đoán</div>
      </div>
      <div className="sample-card__actions">
        <button className="button" onClick={() => onSelect(row)}>Nạp mẫu</button>
      </div>
    </div>
  );

  return (
    <div className="sample-selector">
      <h3>10 ảnh mẫu ngẫu nhiên</h3>
      <p className="sample-selector__note">Bấm lại “Dùng ảnh mẫu” để đổi sang một bộ mẫu ngẫu nhiên khác.</p>

      <div className="sample-group">
        <div className="sample-list-horizontal">{rows.map(renderCard)}</div>
      </div>
    </div>
  );
}

export default SampleSelector;


