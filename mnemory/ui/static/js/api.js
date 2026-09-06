/**
 * mnemory API client — auth-aware fetch wrapper.
 *
 * Stores the API key in localStorage and sends it as X-API-Key
 * on every request. Supports X-User-Id for multi-user switching.
 * Auto-triggers logout on 401 responses.
 */

const MnemoryAPI = {
  /** Base URL for API calls (same origin) */
  baseUrl: '/api',

  /** Get stored API key */
  getKey() {
    return localStorage.getItem('mnemory_api_key') || '';
  },

  /** Store API key */
  setKey(key) {
    localStorage.setItem('mnemory_api_key', key);
  },

  /** Clear stored key */
  clearKey() {
    localStorage.removeItem('mnemory_api_key');
  },

  /** Get the currently selected user_id for switching */
  getSelectedUser() {
    return localStorage.getItem('mnemory_selected_user') || '';
  },

  /** Set the selected user_id for switching */
  setSelectedUser(userId) {
    if (userId) {
      localStorage.setItem('mnemory_selected_user', userId);
    } else {
      localStorage.removeItem('mnemory_selected_user');
    }
  },

  /** Whether auth is via cross-service cookie (no explicit key needed) */
  cookieAuth: false,

  /**
   * Build request headers with auth and identity.
   * When cookieAuth is true, the browser sends the cognis_session cookie
   * automatically — no explicit key header needed.
   */
  _headers(extra = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...extra,
    };
    const key = this.getKey();
    if (key) {
      headers['X-API-Key'] = key;
    }
    const selectedUser = this.getSelectedUser();
    if (selectedUser) {
      headers['X-User-Id'] = selectedUser;
    }
    return headers;
  },

  /**
   * Core fetch wrapper with error handling.
   * Returns parsed JSON on success, throws on error.
   */
  async _fetch(path, options = {}) {
    const url = `${this.baseUrl}${path}`;
    const { headers: extraHeaders, ...rest } = options;
    const response = await fetch(url, {
      ...rest,
      headers: this._headers(extraHeaders),
    });

    if (response.status === 401) {
      // Trigger logout via Alpine store
      if (window.Alpine) {
        const store = Alpine.store('auth');
        if (store) store.logout();
      }
      throw new Error('Unauthorized');
    }

    if (!response.ok) {
      let msg = `HTTP ${response.status}`;
      let detail = null;
      try {
        const body = await response.json();
        detail = body.detail || null;
        msg = typeof detail === 'string'
          ? detail
          : (detail?.message || body.message || body.error || msg);
      } catch { /* ignore parse errors */ }
      const error = new Error(msg);
      error.status = response.status;
      error.detail = detail;
      throw error;
    }

    // Handle empty responses (204 No Content)
    if (response.status === 204) return null;

    return response.json();
  },

  /** GET request */
  async get(path, params = {}) {
    const query = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== null && v !== undefined && v !== '') {
        if (Array.isArray(v)) {
          v.forEach(item => query.append(k, item));
        } else {
          query.set(k, String(v));
        }
      }
    }
    const qs = query.toString();
    return this._fetch(`${path}${qs ? '?' + qs : ''}`);
  },

  /** POST request */
  async post(path, body = {}) {
    return this._fetch(path, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },

  /** PUT request */
  async put(path, body = {}, headers = {}) {
    return this._fetch(path, {
      method: 'PUT',
      body: JSON.stringify(body),
      headers,
    });
  },

  /** DELETE request */
  async del(path, headers = {}) {
    return this._fetch(path, { method: 'DELETE', headers });
  },

  // ── Convenience methods ──────────────────────────────────────

  whoami() {
    return this.get('/whoami');
  },

  stats() {
    return this.get('/stats');
  },

  categories() {
    return this.get('/categories');
  },

  getCoreMemories(params = {}, agentId = '') {
    const query = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== null && v !== undefined && v !== '') {
        query.set(k, String(v));
      }
    }
    const qs = query.toString();
    return this._fetch(`/memories/core${qs ? '?' + qs : ''}`, {
      headers: agentId ? { 'X-Agent-Id': agentId } : {},
    });
  },

  searchMemories(query, filters = {}) {
    return this.post('/memories/search', { query, ...filters });
  },

  findMemories(question, filters = {}) {
    return this.post('/memories/find', { question, ...filters });
  },

  askMemories(question, filters = {}) {
    return this.post('/memories/ask', { question, ...filters });
  },

  listMemories(params = {}) {
    return this.get('/memories', params);
  },

  browseMemories(params = {}) {
    return this.get('/memories/browse', params);
  },

  getMemoriesByIds(ids) {
    return this.post('/memories/by-ids', { ids });
  },

  updateMemory(id, data, revision = null) {
    const headers = revision ? { 'If-Match': `"${revision}"` } : {};
    return this.put(`/memories/${id}`, data, headers);
  },

  deleteMemory(id, revision = null) {
    const headers = revision ? { 'If-Match': `"${revision}"` } : {};
    return this.del(`/memories/${id}`, headers);
  },

  privacyEraseMemory(id) {
    return this.del(`/memories/${id}/privacy`);
  },

  getMemoryHistory(id, params = {}) {
    return this.get(`/memories/${id}/history`, params);
  },

  getMemoryLinks(id) {
    return this.get(`/memories/${id}/links`);
  },

  addMemory(data) {
    return this.post('/memories', data);
  },

  listArtifacts(memoryId) {
    return this.get(`/memories/${memoryId}/artifacts`);
  },

  getArtifact(memoryId, artifactId, offset = 0, limit = 5000) {
    return this.get(`/memories/${memoryId}/artifacts/${artifactId}`, { offset, limit });
  },

  /**
   * Get a signed URL for raw artifact download. Requests a short-lived
   * download token from the server, then returns the URL with the token
   * embedded as a query parameter. Use for <img src="...">, download links.
   *
   * @returns {Promise<string>} Signed URL for direct browser access.
   */
  async getArtifactRawUrl(memoryId, artifactId) {
    const result = await this.post(
      `/memories/${memoryId}/artifacts/${artifactId}/download-token`,
      {},
    );
    if (!result?.url) throw new Error('No download URL returned');
    return result.url;
  },

  saveArtifact(memoryId, data, revision = null) {
    const headers = revision ? { 'If-Match': `"${revision}"` } : {};
    return this._fetch(`/memories/${memoryId}/artifacts`, {
      method: 'POST',
      body: JSON.stringify(data),
      headers,
    });
  },

  deleteArtifact(memoryId, artifactId, revision = null) {
    const headers = revision ? { 'If-Match': `"${revision}"` } : {};
    return this.del(`/memories/${memoryId}/artifacts/${artifactId}`, headers);
  },

  // ── Fsck (Memory Check) ───────────────────────────────────────

  startFsck(params = {}) {
    return this.post('/fsck', params);
  },

  getFsckStatus(checkId) {
    return this.get(`/fsck/${checkId}`);
  },

  applyFsck(checkId, issueIds = null) {
    return this.post(`/fsck/${checkId}/apply`, { issue_ids: issueIds });
  },

  autoRunFsck() {
    return this.post('/fsck/auto-run', {});
  },
};

// ── Color helpers ──────────────────────────────────────────────────
// Read brand colors from CSS custom properties (defined in input.css)
// so JS and CSS stay in sync from a single source of truth.

/**
 * Read a CSS custom property value from :root.
 * @param {string} name - Property name including -- prefix
 * @returns {string} Trimmed property value
 */
function getCSSVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Memory type → color, read from CSS custom properties */
function getTypeColors() {
  return {
    fact:       getCSSVar('--color-mem-fact')       || '#3B82F6',
    preference: getCSSVar('--color-mem-preference') || '#8B5CF6',
    episodic:   getCSSVar('--color-mem-episodic')   || '#F59E0B',
    procedural: getCSSVar('--color-mem-procedural') || '#10B981',
    context:    getCSSVar('--color-mem-context')     || '#64748B',
  };
}

/** Importance level → color, read from CSS custom properties */
function getImportanceColors() {
  return {
    low:      getCSSVar('--color-imp-low')      || '#64748B',
    normal:   getCSSVar('--color-imp-normal')    || '#22D3EE',
    high:     getCSSVar('--color-imp-high')      || '#F59E0B',
    critical: getCSSVar('--color-imp-critical')  || '#EF4444',
  };
}
