export const $ = id => document.getElementById(id);

export const esc = value => String(value ?? '').replace(
  /[&<>'"]/g,
  character => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[character])
);

export const cssEscape = value => globalThis.CSS?.escape
  ? CSS.escape(String(value))
  : String(value).replace(/["\\]/g, '\\$&');

export const optional = id => $(id).value.trim() || null;

export const safeUrl = value => {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : '#';
  } catch {
    return '#';
  }
};

export const formatDateTime = value => {
  if (!value) return '未知时间';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '未知时间' : date.toLocaleString();
};

export function toast(message, error = false) {
  const node = $('toast');
  node.textContent = message;
  node.className = error ? 'show error' : 'show';
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.className = '', 3200);
}

export function busy(form, active) {
  form.classList.toggle('loading', active);
  form.setAttribute('aria-busy', active ? 'true' : 'false');
}

export function tags(items = []) {
  return `<div class="tag-list">${items.map(value => `<span class="tag">${esc(value)}</span>`).join('')}</div>`;
}

export function list(title, items = []) {
  return items.length
    ? `<div class="result-block"><h4>${esc(title)}</h4><ul>${items.map(value => `<li>${esc(value)}</li>`).join('')}</ul></div>`
    : '';
}
