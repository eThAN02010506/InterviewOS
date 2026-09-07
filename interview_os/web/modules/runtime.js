/** Resolve the transport contract injected by a browser or desktop launcher. */

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]']);

function normalizeApiBaseUrl(value, pageLocation = window.location, desktop = false) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  const parsed = new URL(raw, pageLocation.href);
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error('InterviewOS API 地址必须使用 HTTP 或 HTTPS');
  }
  // A desktop sidecar is intentionally local. Browser deployments may still
  // use same-origin or an explicitly injected HTTPS endpoint.
  if (desktop && !LOOPBACK_HOSTS.has(parsed.hostname)) {
    throw new Error('桌面模式只允许连接本机 InterviewOS 服务');
  }
  return parsed.href.replace(/\/$/, '');
}

export function readRuntimeConfig(windowObject = window) {
  const injected = windowObject.__INTERVIEW_OS_RUNTIME__ || {};
  const mode = injected.mode === 'desktop' ? 'desktop' : 'browser';
  const metaBase = windowObject.document
    ?.querySelector('meta[name="interview-os-api-base"]')?.content;
  return Object.freeze({
    apiBaseUrl: normalizeApiBaseUrl(
      injected.apiBaseUrl || metaBase || '', windowObject.location, mode === 'desktop'
    ),
    bootstrapToken: String(injected.bootstrapToken || ''),
    mode
  });
}

export async function loadRuntimeConfig(windowObject = window) {
  const injected = windowObject.__INTERVIEW_OS_RUNTIME__;
  if (injected) return readRuntimeConfig(windowObject);

  const invoke = windowObject.__TAURI__?.core?.invoke;
  if (typeof invoke !== 'function') return readRuntimeConfig(windowObject);

  const desktopRuntime = await invoke('desktop_runtime_config');
  return readRuntimeConfig({
    ...windowObject,
    __INTERVIEW_OS_RUNTIME__: desktopRuntime,
    document: windowObject.document,
    location: windowObject.location
  });
}

export function resolveApiUrl(path, runtime) {
  if (!runtime.apiBaseUrl) return path;
  const normalizedPath = String(path).startsWith('/') ? path : `/${path}`;
  return `${runtime.apiBaseUrl}${normalizedPath}`;
}
