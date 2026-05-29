const RAW_API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.trim();
const DEFAULT_LOCAL_API_BASE = 'http://127.0.0.1:8000';
const DEFAULT_API_BASE = import.meta.env.DEV ? DEFAULT_LOCAL_API_BASE : '';
const API_BASE_LABEL = (RAW_API_BASE && RAW_API_BASE.length > 0 ? RAW_API_BASE : DEFAULT_API_BASE) || 'same-origin';

// Use local backend in dev, but same-origin by default for deployed builds.
export const API_BASE = (RAW_API_BASE && RAW_API_BASE.length > 0 ? RAW_API_BASE : DEFAULT_API_BASE).replace(/\/+$/, '');

export const API_URL = `${API_BASE}/predict`;
export const HEALTH_URL = `${API_BASE}/health`;

export async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = 20000): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: init.signal ?? controller.signal });
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function parseJsonOrText(response: Response): Promise<any> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { __text: text };
  }
}

export function apiPayloadMessage(payload: any, fallback: string): string {
  const detail = payload?.detail;
  const message = detail?.message ?? payload?.message ?? payload?.__text ?? fallback;
  const issues = Array.isArray(detail?.issues) ? detail.issues : [];
  return [message, ...issues].filter(Boolean).join(' ');
}

export function sampleImageUrl(path: string, view = 'preview-v3'): string {
  return `${API_BASE}/sample-image?path=${encodeURIComponent(path)}&view=${encodeURIComponent(view)}`;
}

export function toApiErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof DOMException && error.name === 'AbortError') {
    return `${fallback} API tại ${API_BASE_LABEL} không phản hồi trong thời gian chờ. Backend có thể đang treo hoặc đang khởi động model.`;
  }
  if (error instanceof TypeError) {
    return `${fallback} Không thể kết nối API tại ${API_BASE_LABEL}. Hãy kiểm tra backend đang chạy và cấu hình VITE_API_BASE nếu đang dùng môi trường tùy chỉnh.`;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}
