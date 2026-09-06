/** Resolve the transport contract injected by a browser or desktop launcher. */

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]']);

function normalizeApiBaseUrl(value, pageLocation = window.location) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  const parsed = new URL(raw, pageLocation.href);
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error('InterviewOS API 地址必须使用 HTTP 或 HTTPS');
  }
  // A desktop sidecar is intentionally local. Browser deployments may still
  // use same-origin or an explicitly injected HTTPS endpoint.
  const desktopPage = !['http:', 'https:'].includes(pageLocation.protocol);
  if (desktopPage && !LOOPBACK_HOSTS.has(parsed.hostname)) {
    throw new Error('桌面模式只允许连接本机 InterviewOS 服务');
  }
  return parsed.href.replace(/\/$/, '');
}

export function readRuntimeConfig(windowObject = window) {
  const injected = windowObject.__INTERVIEW_OS_RUNTIME__ || {};
  const metaBase = windowObject.document
    ?.querySelector('meta[name="interview-os-api-base"]')?.content;
  return Object.freeze({
    apiBaseUrl: normalizeApiBaseUrl(injected.apiBaseUrl || metaBase || '', windowObject.location),
    bootstrapToken: String(injected.bootstrapToken || ''),
    mode: injected.mode === 'desktop' ? 'desktop' : 'browser'
  });
}

export function resolveApiUrl(path, runtime) {
  if (!runtime.apiBaseUrl) return path;
  const normalizedPath = String(path).startsWith('/') ? path : `/${path}`;
  return `${runtime.apiBaseUrl}${normalizedPath}`;
}
