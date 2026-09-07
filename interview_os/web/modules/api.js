import {readRuntimeConfig, resolveApiUrl} from './runtime.js';

/** Build the authenticated transport used by every workspace controller. */
export function createApiClient({state, onUnauthorized, runtime = readRuntimeConfig()}) {
  const activeControllers = new Set();
  let globalAbortSignal = null;

  async function request(path, options = {}) {
    const requestToken = state.token;
    const requestSessionId = state.sessionId;
    const authEndpoint = /^\/api\/auth\/(login|register|logout)$/.test(path);
    const assertFresh = () => {
      if (!authEndpoint && (state.token !== requestToken || state.sessionId !== requestSessionId)) {
        throw new DOMException('请求所属的账号或会话已变更，返回结果已忽略', 'AbortError');
      }
    };
    const headers = new Headers(options.headers || {});
    if (!(options.body instanceof FormData) && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }
    // Protected transport headers are assigned last so an individual feature
    // cannot accidentally shadow the desktop or account security boundary.
    if (runtime.bootstrapToken) headers.set('X-InterviewOS-Bootstrap', runtime.bootstrapToken);
    if (requestToken) headers.set('Authorization', `Bearer ${requestToken}`);
    const controller = new AbortController();
    const linkedSignals = [options.signal, globalAbortSignal].filter(Boolean);
    let release = () => {};
    const abortLinkedRequest = event => {
      controller.abort(event.target?.reason);
      release();
    };
    linkedSignals.forEach(signal => {
      if (signal.aborted) controller.abort(signal.reason);
      else signal.addEventListener('abort', abortLinkedRequest, {once: true});
    });
    activeControllers.add(controller);
    let activeReleased = false;
    let linkedSignalsReleased = false;
    release = ({keepLinkedSignals = false} = {}) => {
      if (!activeReleased) {
        activeReleased = true;
        activeControllers.delete(controller);
      }
      if (!keepLinkedSignals && !linkedSignalsReleased) {
        linkedSignalsReleased = true;
        linkedSignals.forEach(signal => signal.removeEventListener('abort', abortLinkedRequest));
      }
    };
    try {
      const response = await fetch(resolveApiUrl(path, runtime), {
        ...options,
        headers,
        signal: controller.signal
      });
      // Responses belong to the authentication/session generation that issued
      // them. Never let a delayed success from account A or candidate A repaint
      // a newly selected workspace.
      assertFresh();
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        assertFresh();
        // Wrong credentials are an expected authentication result. A 401 from
        // any other endpoint means the stored session is no longer valid.
        // A delayed 401 from the previous login must never clear a freshly
        // established session in this tab.
        if (response.status === 401 && !authEndpoint && state.token === requestToken) {
          onUnauthorized();
        }
        const error = new Error(data.detail || `请求失败 (${response.status})`);
        error.status = response.status;
        throw error;
      }
      return {response, assertFresh, release, hasLinkedSignals: linkedSignals.length > 0};
    } catch (error) {
      // If freshness/status validation rejects after response headers arrived,
      // actively stop any unread body instead of relying on eventual GC.
      controller.abort(error);
      release();
      throw error;
    }
  }

  async function api(path, options = {}) {
    const requestContext = await request(path, options);
    try {
      // A successful endpoint promises JSON. In particular, never turn an
      // AbortError while reading a streamed body into an empty success value.
      const data = await requestContext.response.json();
      requestContext.assertFresh();
      return data;
    } finally {
      requestContext.release();
    }
  }

  api.raw = async (path, options = {}) => {
    const requestContext = await request(path, options);
    // Streaming callers keep their explicit AbortSignal linked after headers
    // arrive; ordinary small raw requests can leave the active set immediately.
    requestContext.release({keepLinkedSignals: requestContext.hasLinkedSignals});
    return requestContext.response;
  };
  api.blob = async (path, options = {}) => {
    const requestContext = await request(path, options);
    try {
      const data = await requestContext.response.blob();
      requestContext.assertFresh();
      return data;
    } finally {
      requestContext.release();
    }
  };
  api.text = async (path, options = {}) => {
    const requestContext = await request(path, options);
    try {
      const data = await requestContext.response.text();
      requestContext.assertFresh();
      return data;
    } finally {
      requestContext.release();
    }
  };
  api.runtime = runtime;
  api.abortActive = reason => {
    activeControllers.forEach(controller => controller.abort(reason));
  };
  api.setGlobalAbortSignal = signal => {
    globalAbortSignal = signal || null;
  };
  return api;
}
