import {readRuntimeConfig, resolveApiUrl} from './runtime.js';

/** Build the authenticated transport used by every workspace controller. */
export function createApiClient({state, onUnauthorized, runtime = readRuntimeConfig()}) {
  async function request(path, options = {}) {
    const authHeaders = state.token ? {Authorization: `Bearer ${state.token}`} : {};
    const runtimeHeaders = runtime.bootstrapToken
      ? {'X-InterviewOS-Bootstrap': runtime.bootstrapToken}
      : {};
    const headers = options.body instanceof FormData
      ? {...runtimeHeaders, ...authHeaders, ...(options.headers || {})}
      : {'Content-Type':'application/json', ...runtimeHeaders, ...authHeaders, ...(options.headers || {})};
    const response = await fetch(resolveApiUrl(path, runtime), {...options, headers});
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      // Wrong credentials are an expected authentication result. A 401 from
      // any other endpoint means the stored session is no longer valid.
      const authEndpoint = /^\/api\/auth\/(login|register|logout)$/.test(path);
      if (response.status === 401 && !authEndpoint) onUnauthorized();
      throw new Error(data.detail || `请求失败 (${response.status})`);
    }
    return response;
  }

  async function api(path, options = {}) {
    const response = await request(path, options);
    return response.json().catch(() => ({}));
  }

  api.raw = request;
  api.blob = async (path, options = {}) => (await request(path, options)).blob();
  api.text = async (path, options = {}) => (await request(path, options)).text();
  api.runtime = runtime;
  return api;
}
