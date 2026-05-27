import { useEffect, useMemo, useRef, useState } from 'react';
import { AnalyticsPanel } from './components/AnalyticsPanel';
import { HomePanel } from './components/HomePanel';
import { HistoryPanel } from './components/HistoryPanel';
import { InputPanel } from './components/InputPanel';
import { ResultPanel } from './components/ResultPanel';
import { Sidebar } from './components/Sidebar';
import { API_BASE, API_URL, HEALTH_URL, HISTORY_URL, apiFetch, apiPayloadMessage, parseJsonOrText, toApiErrorMessage } from './api';
import type { FormState, PredictionHistoryRecord, PredictionResponse, ValidationErrors } from './types';

const CT_WARNING_BASE = 'Đây không phải ảnh CT phổi phù hợp cho mô hình. Vui lòng tải đúng ảnh CT phổi.';

type ImageInfo = {
  previewUrl: string | null;
  source: 'sample' | 'upload' | 'none';
  mimeType: string | null;
  width: number | null;
  height: number | null;
  isValid: boolean;
};

type UploadCheckResult = {
  width: number;
  height: number;
  isValid: boolean;
  warnings: string[];
};

type ViewKey = 'home' | 'predict' | 'history';

const FIRST_NAMES = ['Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Vũ', 'Phan', 'Đặng', 'Bùi', 'Đỗ'];
const MIDDLE_NAMES = ['Văn', 'Thị', 'Minh', 'Hữu', 'Ngọc', 'Gia', 'Khánh', 'Anh', 'Quang', 'Thanh'];
const LAST_NAMES = ['An', 'Bình', 'Chi', 'Dũng', 'Hà', 'Hùng', 'Linh', 'Nam', 'Phúc', 'Trang'];

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function generateRandomPatientName(): string {
  return `${pickRandom(FIRST_NAMES)} ${pickRandom(MIDDLE_NAMES)} ${pickRandom(LAST_NAMES)}`;
}

function dequote(value: string): string {
  const v = value.trim();
  if (v.length >= 2 && v.startsWith('"') && v.endsWith('"')) {
    return v.slice(1, -1);
  }
  return v;
}

function parseCsvRows(text: string): Record<string, string>[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length === 0) return [];
  const header = lines.shift()!.split(',').map(dequote);
  return lines.map((line) => {
    const parts = line.split(',').map(dequote);
    const row: Record<string, string> = {};
    header.forEach((h, i) => {
      row[h] = parts[i] ?? '';
    });
    return row;
  });
}

function isImageLike(input: { type?: string; name?: string }): boolean {
  const type = (input.type ?? '').toLowerCase();
  const name = (input.name ?? '').toLowerCase();
  if (type.startsWith('image/')) return true;
  return ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff'].some((ext) => name.endsWith(ext));
}

async function readImageDimensions(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error('load-failed'));
    img.src = url;
  });
}

async function analyzeUploadedImage(file: File, url: string): Promise<UploadCheckResult> {
  const warnings: string[] = [];
  const blockingIssues: string[] = [];
  const img = new Image();
  const addBlock = (msg: string) => {
    warnings.push(msg);
    blockingIssues.push(msg);
  };
  const addWarn = (msg: string) => {
    warnings.push(msg);
  };

  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('Ảnh bị hỏng hoặc không đọc được.'));
    img.src = url;
  });

  const width = img.naturalWidth || 0;
  const height = img.naturalHeight || 0;
  if (width <= 0 || height <= 0) {
    return {
      width,
      height,
      isValid: false,
      warnings: ['Ảnh không đọc được kích thước hợp lệ.'],
    };
  }

  if (width < 96 || height < 96) {
    addBlock('Ảnh quá nhỏ cho mô hình dự đoán (khuyến nghị tối thiểu 96x96).');
  }

  const ratio = width / Math.max(1, height);
  if (ratio < 0.45 || ratio > 2.2) {
    addWarn('Tỉ lệ ảnh bất thường, không giống lát cắt CT phổi điển hình.');
  }

  const canvas = document.createElement('canvas');
  const maxSide = 256;
  const scale = Math.min(1, maxSide / Math.max(width, height));
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return {
      width,
      height,
      isValid: false,
      warnings: ['Không thể phân tích nhanh ảnh đầu vào.'],
    };
  }
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

  const pxCount = canvas.width * canvas.height;
  const lumArr = new Float32Array(pxCount);
  const bodyMask = new Uint8Array(pxCount);
  const airMask = new Uint8Array(pxCount);
  let grayDevSum = 0;
  let lumSum = 0;
  let lumSqSum = 0;
  let n = 0;
  let bodyCount = 0;
  let airCount = 0;
  let bodyLeft = 0;
  let bodyRight = 0;
  let airLeft = 0;
  let airRight = 0;
  const midX = canvas.width / 2;

  for (let i = 0; i < data.length; i += 4) {
    const p = i / 4;
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    grayDevSum += (Math.abs(r - g) + Math.abs(g - b) + Math.abs(r - b)) / 3;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    lumArr[p] = lum;
    lumSum += lum;
    lumSqSum += lum * lum;
    n += 1;

    const x = p % canvas.width;
    const isBody = lum > 12;
    if (isBody) {
      bodyMask[p] = 1;
      bodyCount += 1;
      if (x < midX) bodyLeft += 1;
      else bodyRight += 1;
    }

    const isAir = isBody && lum < 70;
    if (isAir) {
      airMask[p] = 1;
      airCount += 1;
      if (x < midX) airLeft += 1;
      else airRight += 1;
    }
  }

  if (n > 0) {
    const grayDev = grayDevSum / n; // 0..255
    const meanLum = lumSum / n;
    const stdLum = Math.sqrt(Math.max(0, lumSqSum / n - meanLum * meanLum));

    if (grayDev > 14) {
      addBlock('Ảnh không giống grayscale/CT-like (màu sắc khác biệt cao).');
    }
    if (stdLum < 14) {
      addWarn('Ảnh thiếu tương phản, có thể không phải lát cắt CT phù hợp.');
    }
    if (meanLum < 20 || meanLum > 235) {
      addWarn('Ảnh quá tối hoặc quá sáng so với miền dữ liệu huấn luyện.');
    }
  }

  const bodyRatio = bodyCount / Math.max(1, pxCount);
  if (bodyRatio < 0.12 || bodyRatio > 0.95) {
    addWarn('Vùng cơ thể trong ảnh không hợp lý cho lát cắt CT phổi.');
  }

  const airRatioInBody = airCount / Math.max(1, bodyCount);
  if (airRatioInBody < 0.17 || airRatioInBody > 0.82) {
    addBlock('Phân bố vùng khí không giống CT phổi (thiếu cấu trúc phổi điển hình).');
  }

  const leftAirRatio = airLeft / Math.max(1, bodyLeft);
  const rightAirRatio = airRight / Math.max(1, bodyRight);
  if (leftAirRatio < 0.08 || rightAirRatio < 0.08) {
    addBlock('Không thấy đủ vùng khí ở cả hai bên phổi.');
  }
  const lrBalance = Math.min(leftAirRatio, rightAirRatio) / Math.max(1e-6, Math.max(leftAirRatio, rightAirRatio));
  if (lrBalance < 0.35) {
    addBlock('Hai vùng khí trái/phải mất cân đối mạnh, không giống CT phổi điển hình.');
  }

  const visited = new Uint8Array(pxCount);
  const largeAirComponents: Array<{ area: number; cx: number; cy: number }> = [];
  const minCompArea = Math.max(40, Math.floor(pxCount * 0.008));
  const qx = new Int32Array(pxCount);
  const qy = new Int32Array(pxCount);

  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const idx = y * canvas.width + x;
      if (!airMask[idx] || visited[idx]) continue;
      let head = 0;
      let tail = 0;
      qx[tail] = x;
      qy[tail] = y;
      tail += 1;
      visited[idx] = 1;
      let area = 0;
      let sumX = 0;
      let sumY = 0;

      while (head < tail) {
        const cx = qx[head];
        const cy = qy[head];
        head += 1;
        area += 1;
        sumX += cx;
        sumY += cy;

        const neighbors = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1],
        ];
        for (const [nx, ny] of neighbors) {
          if (nx < 0 || ny < 0 || nx >= canvas.width || ny >= canvas.height) continue;
          const nIdx = ny * canvas.width + nx;
          if (!airMask[nIdx] || visited[nIdx]) continue;
          visited[nIdx] = 1;
          qx[tail] = nx;
          qy[tail] = ny;
          tail += 1;
        }
      }

      if (area >= minCompArea) {
        largeAirComponents.push({
          area,
          cx: sumX / area,
          cy: sumY / area,
        });
      }
    }
  }

  const lungsLike = largeAirComponents
    .sort((a, b) => b.area - a.area)
    .slice(0, 4)
    .filter((c) => c.cy > canvas.height * 0.12 && c.cy < canvas.height * 0.88);
  const hasLeft = lungsLike.some((c) => c.cx < canvas.width * 0.46);
  const hasRight = lungsLike.some((c) => c.cx > canvas.width * 0.54);
  if (!(hasLeft && hasRight)) {
    addBlock('Không nhận diện được hai vùng phổi trái/phải đặc trưng.');
  }

  let boundaryCount = 0;
  let brightBoundaryCount = 0;
  for (let y = 1; y < canvas.height - 1; y++) {
    for (let x = 1; x < canvas.width - 1; x++) {
      const idx = y * canvas.width + x;
      if (!bodyMask[idx]) continue;
      const up = bodyMask[idx - canvas.width];
      const down = bodyMask[idx + canvas.width];
      const left = bodyMask[idx - 1];
      const right = bodyMask[idx + 1];
      if (!(up && down && left && right)) {
        boundaryCount += 1;
        if (lumArr[idx] > 205) brightBoundaryCount += 1;
      }
    }
  }
  if (boundaryCount > 0) {
    const brightBoundaryRatio = brightBoundaryCount / boundaryCount;
    if (brightBoundaryRatio > 0.33) {
      addBlock('Biên sáng dạng vòng cao, nghiêng về CT sọ não hoặc vùng khác.');
    }
  }

  const ext = (file.name.split('.').pop() ?? '').toLowerCase();
  const allowedExt = new Set(['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp']);
  if (ext && !allowedExt.has(ext)) {
    addWarn('Định dạng ảnh không thuộc nhóm thường dùng cho dự đoán CT.');
  }

  return {
    width,
    height,
    isValid: blockingIssues.length === 0,
    warnings,
  };
}

const initialForm: FormState = {
  image: null,
  imagePreview: null,
  sample_path: null,
  localization_sample_path: null,
  patient_name: '',
  age: '',
  sex: '',
  smoking_status: '',
  tumor_size: '',
  family_history: '',
  symptom_score: '',
};

function validate(form: FormState): ValidationErrors {
  const errors: ValidationErrors = {};

  if (!form.image && !form.sample_path) {
    errors.image = 'Vui lòng chọn ảnh CT (upload hoặc ảnh mẫu).';
  } else if (form.image && !isImageLike(form.image)) {
    errors.image = 'File đầu vào không phải định dạng ảnh hợp lệ.';
  }
  const age = Number(form.age);
  const tumorSizeText = form.tumor_size.trim();
  const tumorSize = Number(tumorSizeText);
  const symptomScore = Number(form.symptom_score);

  if (!Number.isFinite(age) || age < 20 || age > 90) errors.age = 'Tuổi phải trong khoảng 20-90.';
  if (tumorSizeText && (!Number.isFinite(tumorSize) || tumorSize < 2 || tumorSize > 40)) {
    errors.tumor_size = 'Kích thước nốt ác tính phải trong khoảng 2-40 mm, hoặc để trống nếu chưa có thông tin.';
  }
  if (!Number.isFinite(symptomScore) || symptomScore < 1 || symptomScore > 10) errors.symptom_score = 'Mức độ triệu chứng phải trong khoảng 1-10.';

  if (!form.sex) errors.sex = 'Vui lòng chọn giới tính.';
  if (!form.smoking_status) errors.smoking_status = 'Vui lòng chọn tình trạng hút thuốc.';
  if (!form.family_history) errors.family_history = 'Vui lòng chọn tiền sử gia đình.';
  if (!form.patient_name.trim()) errors.patient_name = 'Vui lòng nhập tên bệnh nhân.';

  return errors;
}

export default function App() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({});
  const [sampleLoaded, setSampleLoaded] = useState(false);
  const [historyItems, setHistoryItems] = useState<PredictionHistoryRecord[]>([]);
  const [historySelected, setHistorySelected] = useState<PredictionHistoryRecord | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<ViewKey>('home');
  const [imageInfo, setImageInfo] = useState<ImageInfo>({
    previewUrl: null,
    source: 'none',
    mimeType: null,
    width: null,
    height: null,
    isValid: false,
  });
  const lastObjectUrl = useRef<string | null>(null);

  const onChange = (patch: Partial<FormState>) => {
    setForm((current) => ({ ...current, ...patch }));
    setValidationErrors((current) => ({ ...current, ...patchValidationErrors(patch) }));
  };

  const patchValidationErrors = (patch: Partial<FormState>): ValidationErrors => {
    const next: ValidationErrors = {};
    if ('patient_name' in patch) next.patient_name = undefined;
    if ('age' in patch) next.age = undefined;
    if ('sex' in patch) next.sex = undefined;
    if ('smoking_status' in patch) next.smoking_status = undefined;
    if ('tumor_size' in patch) next.tumor_size = undefined;
    if ('family_history' in patch) next.family_history = undefined;
    if ('symptom_score' in patch) next.symptom_score = undefined;
    return next;
  };

  const onFileChange = async (file: File | null) => {
    setError(null);
    if (!file) {
      if (lastObjectUrl.current && lastObjectUrl.current.startsWith('blob:')) {
        try { URL.revokeObjectURL(lastObjectUrl.current); } catch {}
      }
      lastObjectUrl.current = null;
      setForm((current) => ({ ...current, image: null, imagePreview: null, sample_path: null, localization_sample_path: null }));
      setValidationErrors((current) => ({ ...current, image: undefined }));
      setSampleLoaded(false);
      setImageInfo({ previewUrl: null, source: 'none', mimeType: null, width: null, height: null, isValid: false });
      return;
    }
    if (!isImageLike(file)) {
      setValidationErrors((current) => ({ ...current, image: CT_WARNING_BASE }));
      setError(CT_WARNING_BASE);
      return;
    }

    if (lastObjectUrl.current && lastObjectUrl.current.startsWith('blob:')) {
      try { URL.revokeObjectURL(lastObjectUrl.current); } catch {}
    }

    const imagePreview = URL.createObjectURL(file);
    lastObjectUrl.current = imagePreview;
    setForm((current) => ({ ...current, image: file, imagePreview, sample_path: null, localization_sample_path: null }));
    setValidationErrors((current) => ({ ...current, image: 'Đang kiểm tra ảnh đầu vào...' }));
    setResult(null);
    setSampleLoaded(true);
    setImageInfo({
      previewUrl: imagePreview,
      source: 'upload',
      mimeType: file.type || null,
      width: null,
      height: null,
      isValid: false,
    });

    try {
      const check = await analyzeUploadedImage(file, imagePreview);
      setImageInfo((current) =>
        current.previewUrl === imagePreview
          ? { ...current, width: check.width, height: check.height, isValid: check.isValid }
          : current,
      );

      if (!check.isValid) {
        const detail = check.warnings.join(' ');
        setValidationErrors((current) => ({
          ...current,
          image: `${CT_WARNING_BASE} Chi tiết: ${detail}`,
        }));
      } else {
        setValidationErrors((current) => ({ ...current, image: undefined }));
      }
    } catch {
      setImageInfo((current) =>
        current.previewUrl === imagePreview
          ? { ...current, width: null, height: null, isValid: false }
          : current,
      );
      setValidationErrors((current) => ({
        ...current,
        image: `${CT_WARNING_BASE} Ảnh bị hỏng hoặc không đọc được.`,
      }));
    }
  };

  const onSampleSelect = async (row: any) => {
    try {
      setError(null);
      const previewPath = String(row.preview_image_path ?? '').trim() || String(row.image_path ?? '').trim();
      const endpoint = `/sample-previews/${encodeURIComponent(String(row.patient_id ?? 'sample'))}.png`;

      // Load sample metadata and keep sample_path for backend predict (no Blob/File conversion needed).
      setForm((current) => ({
        ...current,
        image: null,
        imagePreview: endpoint,
        sample_path: String(row.image_path ?? '').trim() || null,
        localization_sample_path: previewPath || String(row.image_path ?? '').trim() || null,
        patient_name: generateRandomPatientName(),
        age: String(row.age ?? initialForm.age),
        sex: (row.sex as FormState['sex']) ?? initialForm.sex,
        smoking_status: (row.smoking_status as FormState['smoking_status']) ?? initialForm.smoking_status,
        tumor_size: '',
        family_history: (row.family_history as FormState['family_history']) ?? initialForm.family_history,
        symptom_score: String(row.symptom_score ?? initialForm.symptom_score),
      }));
      setValidationErrors((current) => ({
        ...current,
        image: undefined,
        age: undefined,
        sex: undefined,
        smoking_status: undefined,
        tumor_size: undefined,
        family_history: undefined,
        symptom_score: undefined,
      }));
      setSampleLoaded(true);
      setImageInfo({
        previewUrl: endpoint,
        source: 'sample',
        mimeType: null,
        width: null,
        height: null,
        isValid: true,
      });
      readImageDimensions(endpoint)
        .then((dims) => {
          setImageInfo((current) =>
            current.previewUrl === endpoint
              ? { ...current, width: dims.width, height: dims.height }
              : current,
          );
        })
        .catch(() => {});
      setError(null);
      setResult(null);
      setSampleLoaded(true);
    } catch (err) {
      console.error('Sample load failed', err);
      setForm((current) => ({ ...current, image: null, sample_path: null, localization_sample_path: null }));
      setValidationErrors((current) => ({
        ...current,
        image: 'Không thể nạp ảnh mẫu. Hãy chọn lại mẫu hoặc tải ảnh lên.',
      }));
      setError(err instanceof Error ? err.message : 'Không thể nạp ảnh mẫu.');
      setSampleLoaded(false);
      setImageInfo((current) => ({ ...current, isValid: false }));
    }
  };

  const onRandomSample = async () => {
    try {
      const url = new URL('./data/demo_samples_3class.csv', import.meta.url).toString();
      const txt = await fetch(url).then((r) => r.text());
      const rows = parseCsvRows(txt);
      if (rows.length === 0) return;
      const idx = Math.floor(Math.random() * rows.length);
      const row = rows[idx];
      await onSampleSelect(row);
    } catch (e) {
      console.error('random sample failed', e);
    }
  };

  const onRandomizeParams = () => {
    const rand = (min:number, max:number, step=1) => Math.round((Math.random()*(max-min)+min)/step)*step;
    const sexes: FormState['sex'][] = ['female','male'];
    const smokes: FormState['smoking_status'][] = ['never','former','current'];
    const fam: FormState['family_history'][] = ['no','yes'];
    onChange({
      patient_name: generateRandomPatientName(),
      age: String(rand(20,90,1)),
      sex: sexes[Math.floor(Math.random()*sexes.length)],
      smoking_status: smokes[Math.floor(Math.random()*smokes.length)],
      family_history: fam[Math.floor(Math.random()*fam.length)],
      symptom_score: String(rand(1,10,1)),
    });
  };

  const onPredict = async () => {
    setError(null);

    try {
      const healthResp = await apiFetch(HEALTH_URL, {}, 25000);
      if (!healthResp.ok) {
        setError(`Backend không sẵn sàng tại ${API_BASE} (health: ${healthResp.status}). Kiểm tra server FastAPI và route /health.`);
        return;
      }
    } catch {
      setError(`Không thể kết nối backend (${API_BASE}). Hãy kiểm tra API server ở cổng 8000 và cấu hình VITE_API_BASE.`);
      return;
    }

    const samplePath = form.sample_path?.trim() || null;
    const imageForPredict: File | null = form.image;
    if (!imageForPredict && !samplePath) {
      setValidationErrors((current) => ({
        ...current,
        image: 'Không tìm thấy ảnh hợp lệ để dự đoán. Hãy chọn ảnh mẫu hoặc upload ảnh.',
      }));
      setError('Không tìm thấy ảnh hợp lệ để dự đoán. Vui lòng nạp lại ảnh mẫu hoặc tải ảnh lên.');
      return;
    }

    if (imageForPredict && !isImageLike(imageForPredict)) {
      setValidationErrors((current) => ({ ...current, image: CT_WARNING_BASE }));
      setError(CT_WARNING_BASE);
      return;
    }

    if (imageForPredict && imageInfo.source === 'upload' && !imageInfo.isValid) {
      setValidationErrors((current) => ({
        ...current,
        image: current.image ?? CT_WARNING_BASE,
      }));
      setError(CT_WARNING_BASE);
      return;
    }

    setValidationErrors((current) => ({ ...current, image: undefined }));

    const workingForm: FormState = {
      ...form,
      image: imageForPredict,
      sample_path: samplePath,
    };

    const nextErrors = validate(workingForm);
    setValidationErrors(nextErrors);
    if (Object.values(nextErrors).some(Boolean)) {
      setError('Vui lòng kiểm tra lại dữ liệu đầu vào.');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const body = new FormData();
      if (imageForPredict) {
        body.append('image', imageForPredict);
      } else if (samplePath) {
        body.append('sample_path', samplePath);
        if (workingForm.localization_sample_path) {
          body.append('localization_sample_path', workingForm.localization_sample_path);
        }
      }
      body.append('age', workingForm.age);
      body.append('patient_name', workingForm.patient_name);
      body.append('sex', workingForm.sex);
      body.append('smoking_status', workingForm.smoking_status);
      body.append('tumor_size', workingForm.tumor_size);
      body.append('family_history', workingForm.family_history);
      body.append('symptom_score', workingForm.symptom_score);

      const response = await apiFetch(API_URL, { method: 'POST', body }, 45000);
      const payload = await parseJsonOrText(response);

      if (!response.ok) {
        setError(apiPayloadMessage(payload, `Dự đoán thất bại (mã ${response.status})`));
        return;
      }

      setResult(payload as PredictionResponse);
      await fetchHistory();
    } catch (fetchError) {
      setError(toApiErrorMessage(fetchError, 'Predict thất bại.'));
    } finally {
      setLoading(false);
    }
  };

  const onReset = () => {
    if (lastObjectUrl.current && lastObjectUrl.current.startsWith('blob:')) {
      try { URL.revokeObjectURL(lastObjectUrl.current); } catch {}
    }
    lastObjectUrl.current = null;
    setForm(initialForm);
    setResult(null);
    setError(null);
    setValidationErrors({});
    setSampleLoaded(false);
    setImageInfo({ previewUrl: null, source: 'none', mimeType: null, width: null, height: null, isValid: false });
  };

  const fetchHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const response = await apiFetch(`${HISTORY_URL}?limit=100`, {}, 20000);
      const payload = await parseJsonOrText(response);
      if (!response.ok) {
        throw new Error(apiPayloadMessage(payload, 'Không tải được lịch sử dự đoán.'));
      }
      const items = (payload?.items ?? []) as PredictionHistoryRecord[];
      setHistoryItems(items);
      if (historySelected) {
        const keep = items.find((x) => x.id === historySelected.id) ?? null;
        setHistorySelected(keep);
      }
    } catch (e) {
      setHistoryError(toApiErrorMessage(e, 'Không tải được lịch sử dự đoán.'));
    } finally {
      setHistoryLoading(false);
    }
  };

  const onSelectHistory = async (id: string) => {
    setHistoryError(null);
    try {
      const response = await apiFetch(`${HISTORY_URL}/${id}`, {}, 20000);
      const payload = await parseJsonOrText(response);
      if (!response.ok) {
        throw new Error(apiPayloadMessage(payload, 'Không tải được chi tiết lịch sử.'));
      }
      setHistorySelected(payload as PredictionHistoryRecord);
    } catch (e) {
      setHistoryError(toApiErrorMessage(e, 'Không tải được chi tiết lịch sử.'));
    }
  };

  const onDeleteHistory = async (id: string) => {
    setHistoryError(null);
    try {
      const response = await apiFetch(`${HISTORY_URL}/${id}`, { method: 'DELETE' }, 20000);
      const payload = await parseJsonOrText(response);
      if (!response.ok) {
        throw new Error(apiPayloadMessage(payload, 'Không xoá được bản ghi lịch sử.'));
      }
      if (historySelected?.id === id) {
        setHistorySelected(null);
      }
      await fetchHistory();
    } catch (e) {
      setHistoryError(toApiErrorMessage(e, 'Không xoá được bản ghi lịch sử.'));
    }
  };

  useEffect(() => {
    void fetchHistory();
  }, []);

  const onNavigate = (view: ViewKey) => {
    setActiveView(view);
    if (view === 'history') {
      void fetchHistory();
    }
  };

  return (
    <div className="dashboard-shell">
      <Sidebar activeView={activeView} historyCount={historyItems.length} onNavigate={onNavigate} />

      <div className="app-shell">
        <header className="hero">
          <div>
            <h1>
              {activeView === 'home'
                ? 'Trang chủ hệ thống'
                : activeView === 'predict'
                  ? 'Dự đoán nguy cơ ác tính phổi'
                  : 'Lịch sử dự đoán'}
            </h1>
          </div>
          <div className="hero__badge">Chỉ dành nghiên cứu</div>
        </header>

        {activeView === 'home' ? (
          <HomePanel />
        ) : activeView === 'predict' ? (
          <>
            <main className="layout layout--merged">
              <InputPanel
                form={form}
                errors={validationErrors}
                onChange={onChange}
                onFileChange={onFileChange}
                onPredict={onPredict}
                onReset={onReset}
                disabled={loading}
                onSampleSelect={onSampleSelect}
                sampleLoaded={sampleLoaded}
                onRandomSample={onRandomSample}
                onRandomizeParams={onRandomizeParams}
              />

              <ResultPanel loading={loading} error={error} result={result} form={form} imageInfo={imageInfo} />
            </main>
          </>
        ) : (
          <main className="history-page">
            <HistoryPanel
              items={historyItems}
              selected={historySelected}
              loading={historyLoading}
              error={historyError}
        onRefresh={fetchHistory}
        onSelect={onSelectHistory}
        onCollapse={() => setHistorySelected(null)}
        onDelete={onDeleteHistory}
      />
          </main>
        )}
      </div>
    </div>
  );
}

