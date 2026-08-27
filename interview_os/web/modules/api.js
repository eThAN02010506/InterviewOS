/** Build the authenticated JSON client used by every workspace controller. */
export function createApiClient({state, onUnauthorized}) {
  return async function api(path, options = {}) {
    const authHeaders = state.token ? {Authorization: `Bearer ${state.token}`} : {};
    const headers = options.body instanceof FormData
      ? {...authHeaders, ...(options.headers || {})}
      : {'Content-Type':'application/json', ...authHeaders, ...(options.headers || {})};
    const response = await fetch(path, {...options, headers});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      // Wrong credentials are an expected authentication result. A 401 from
      // any other endpoint means the stored session is no longer valid.
      const authEndpoint = /^\/api\/auth\/(login|register|logout)$/.test(path);
      if (response.status === 401 && !authEndpoint) onUnauthorized();
      throw new Error(data.detail || `请求失败 (${response.status})`);
    }
    return data;
  };
}
