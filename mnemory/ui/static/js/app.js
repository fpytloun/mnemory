/**
 * mnemory UI — Alpine.js stores and app initialization.
 *
 * Stores:
 * - auth: login state, API key, identity, user switching
 * - nav: active tab navigation
 * - notify: toast notification system
 */

function trapModalFocus(event) {
  const focusable = [...event.currentTarget.querySelectorAll(
    'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex="0"]'
  )].filter((element) => element.offsetParent !== null);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

document.addEventListener('alpine:init', () => {

  // ── Auth Store ───────────────────────────────────────────────
  Alpine.store('auth', {
    authenticated: false,
    loading: true,
    error: '',
    identity: null,     // { user_id, agent_id, timezone, can_switch_user }
    users: [],          // available users for switching
    selectedUser: '',   // currently selected user (for switching)

    async init() {
      // Check for exchange token in URL (Cognis cross-service SSO)
      const params = new URLSearchParams(window.location.search);
      const exchangeToken = params.get('token');
      if (exchangeToken) {
        try {
          const resp = await fetch('/api/auth/exchange', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: exchangeToken })
          });
          if (resp.ok) {
            // Clean token from URL after successful exchange
            window.history.replaceState({}, '', window.location.pathname);
            // Cookie set by response — proceed with cookie auth
            await this.tryCookieAuth();
            if (this.authenticated) return;
          }
        } catch (e) { console.warn('Exchange token auth failed, falling back:', e); }
      }
      // Check if we have a stored key
      const key = MnemoryAPI.getKey();
      if (key) {
        await this.verify();
      } else {
        // Try cookie-based auth (cross-service SSO from Cognis)
        await this.tryCookieAuth();
      }
    },

    async tryCookieAuth() {
      try {
        const identity = await MnemoryAPI.whoami();
        if (identity && identity.user_id) {
          MnemoryAPI.cookieAuth = true;
          this.identity = identity;
          this.authenticated = true;
          this.selectedUser = identity.user_id;
          if (identity.can_switch_user) {
            await this.loadUsers();
          }
          this.loading = false;
          return;
        }
      } catch {
        // Cookie auth not available — show login form
      }
      this.loading = false;
    },

    async login(apiKey) {
      this.error = '';
      this.loading = true;
      MnemoryAPI.setKey(apiKey);
      try {
        const identity = await MnemoryAPI.whoami();
        this.identity = identity;
        this.authenticated = true;
        // Load users list if switching is allowed (wildcard key)
        if (identity.can_switch_user) {
          await this.loadUsers();
          // If no user_id bound by key, auto-select first known user
          if (!identity.user_id && this.users.length > 0) {
            this.switchUser(this.users[0]);
          } else {
            this.selectedUser = identity.user_id || '';
            MnemoryAPI.setSelectedUser(''); // no override needed
          }
        } else {
          this.selectedUser = identity.user_id || '';
          MnemoryAPI.setSelectedUser(''); // bound key, no override needed
        }
      } catch (e) {
        MnemoryAPI.clearKey();
        this.authenticated = false;
        this.error = e.message === 'Unauthorized'
          ? 'Invalid API key'
          : `Connection failed: ${e.message}`;
      } finally {
        this.loading = false;
      }
    },

    async verify() {
      this.loading = true;
      try {
        const identity = await MnemoryAPI.whoami();
        this.identity = identity;
        this.authenticated = true;
        if (identity.can_switch_user) {
          await this.loadUsers();
          const stored = MnemoryAPI.getSelectedUser();
          if (stored) {
            this.selectedUser = stored;
          } else if (identity.user_id) {
            this.selectedUser = identity.user_id;
          } else if (this.users.length > 0) {
            this.switchUser(this.users[0]);
          }
        } else {
          this.selectedUser = identity.user_id || '';
        }
      } catch {
        MnemoryAPI.clearKey();
        MnemoryAPI.setSelectedUser('');
        this.authenticated = false;
      } finally {
        this.loading = false;
      }
    },

    async loadUsers() {
      try {
        const stats = await MnemoryAPI.stats();
        this.users = stats.users || [];
      } catch {
        // Non-critical — user list just won't be available
        this.users = [];
      }
    },

    switchUser(userId) {
      this.selectedUser = userId;
      if (this.identity && userId === this.identity.user_id) {
        MnemoryAPI.setSelectedUser(''); // use default
      } else {
        MnemoryAPI.setSelectedUser(userId);
      }
      // Dispatch event so tabs can refresh
      window.dispatchEvent(new CustomEvent('mnemory:user-changed', {
        detail: { userId }
      }));
    },

    logout() {
      MnemoryAPI.clearKey();
      MnemoryAPI.setSelectedUser('');
      this.authenticated = false;
      this.identity = null;
      this.users = [];
      this.selectedUser = '';
      this.error = '';
    },
  });

  // ── Memory Edit Store ────────────────────────────────────────
  // Global edit modal — shared between Memories and Search tabs.
  Alpine.store('memoryEdit', {
    open: false,
    memory: null,   // original memory object
    form: {},       // editable copy
    _onSaved: null, // callback(updatedFields) after successful save
    _returnFocus: null,

    /**
     * Open the edit modal for a memory.
     * @param {object} memory - The memory object to edit.
     * @param {function} onSaved - Called with updated field map after save.
     */
    show(memory, onSaved, returnFocus = null) {
      const meta = memory.metadata || {};
      this.memory = memory;
      this._onSaved = onSaved || null;
      this._returnFocus = returnFocus || document.activeElement;
      this.form = {
        content: memory.memory || '',
        memory_type: meta.memory_type || '',
        categories: (meta.categories || []).join(', '),
        importance: meta.importance || 'normal',
        pinned: !!meta.pinned,
        ttl_days: meta.ttl_days ?? '',
        event_date: meta.event_date ? meta.event_date.slice(0, 10) : '',
        agent_id: meta.agent_id || '',
        labels: JSON.stringify(meta.labels || {}, null, 0),
      };
      this.open = true;
      setTimeout(() => {
        document.querySelector('[data-memory-edit-modal] textarea')?.focus();
      }, 200);
    },

    close() {
      this.open = false;
      this.memory = null;
      this._onSaved = null;
      requestAnimationFrame(() => this._returnFocus?.focus());
      this._returnFocus = null;
    },

    trapFocus(event) {
      trapModalFocus(event);
    },

    async save() {
      const form = this.form;
      const memoryId = this.memory.id;
      const meta = this.memory.metadata || {};
      const payload = {};

      if (form.content !== this.memory.memory) payload.content = form.content;
      if (form.memory_type && form.memory_type !== meta.memory_type) payload.memory_type = form.memory_type;

      const newCats = form.categories
        ? form.categories.split(',').map(c => c.trim()).filter(Boolean)
        : [];
      if (JSON.stringify(newCats) !== JSON.stringify(meta.categories || [])) payload.categories = newCats;

      if (form.importance && form.importance !== meta.importance) payload.importance = form.importance;
      if (form.pinned !== !!meta.pinned) payload.pinned = form.pinned;

      if (form.ttl_days !== '' && form.ttl_days !== null) {
        const ttl = parseInt(form.ttl_days, 10);
        if (!isNaN(ttl) && ttl !== meta.ttl_days) payload.ttl_days = ttl;
      }

      const origDate = meta.event_date ? meta.event_date.slice(0, 10) : '';
      if (form.event_date !== origDate) {
        payload.event_date = form.event_date || null;
      }

      // agent_id: empty string = clear, non-empty = set, unchanged = omit
      const origAgent = meta.agent_id || '';
      if (form.agent_id !== origAgent) {
        payload.agent_id = form.agent_id || '';  // '' signals "clear" to the API
      }

      // Labels: compare JSON
      try {
        const newLabels = form.labels ? JSON.parse(form.labels) : {};
        const oldLabels = meta.labels || {};
        if (JSON.stringify(newLabels) !== JSON.stringify(oldLabels)) payload.labels = newLabels;
      } catch (e) {
        // Invalid JSON — skip labels update
      }

      if (Object.keys(payload).length === 0) { this.close(); return; }

      try {
        const result = await MnemoryAPI.updateMemory(memoryId, payload, meta.revision);
        if (this._onSaved) this._onSaved(result);
        Alpine.store('notify').success('A successor revision was created');
        this.close();
      } catch (err) {
        if (err.status === 409) {
          window.dispatchEvent(new CustomEvent('mnemory:stale-revision', {
            detail: err.detail || {},
          }));
          Alpine.store('notify').warning('A newer revision exists. The view was refreshed.');
          this.close();
          return;
        }
        Alpine.store('notify').error(`Failed to update: ${err.message}`);
      }
    },
  });

  // ── Artifact Manager Store ───────────────────────────────────
  // Global artifact modal — shared between Memories and Search tabs.
  Alpine.store('artifactMgr', {
    open: false,
    _returnFocus: null,
    memoryId: null,
    revision: null,
    memoryText: '',
    artifacts: [],
    loading: false,
    deleteConfirm: null,

    // View sub-panel
    view: {
      open: false,
      artifact: null,
      content: '',
      hasMore: false,
      offset: 0,
      loading: false,
    },

    // Add artifact sub-form
    addForm: {
      open: false,
      saving: false,
      filename: 'note.md',
      content_type: 'text/markdown',
      content: '',
    },

    /**
     * Open the artifact manager for a memory.
     * @param {object} memory - Memory object with id and memory text.
     */
    show(memory, returnFocus = null) {
      this._returnFocus = returnFocus || document.activeElement;
      this.memoryId = memory.id;
      this.revision = memory.metadata?.revision || null;
      this.memoryText = memory.memory || '';
      this.artifacts = [];
      this.deleteConfirm = null;
      this.view = { open: false, artifact: null, content: '', hasMore: false, offset: 0, loading: false };
      this.addForm = { open: false, saving: false, filename: 'note.md', content_type: 'text/markdown', content: '' };
      this.open = true;
      setTimeout(() => {
        document.querySelector('[data-artifact-modal] button')?.focus();
      }, 200);
      this.loadArtifacts();
    },

    close() {
      this.open = false;
      this.memoryId = null;
      this.revision = null;
      requestAnimationFrame(() => this._returnFocus?.focus());
      this._returnFocus = null;
    },

    trapFocus(event) {
      trapModalFocus(event);
    },

    async loadArtifacts() {
      this.loading = true;
      try {
        const result = await MnemoryAPI.listArtifacts(this.memoryId);
        // API returns array directly or { artifacts: [...] }
        this.artifacts = Array.isArray(result) ? result : (result.artifacts || []);
      } catch (err) {
        Alpine.store('notify').error(`Failed to load artifacts: ${err.message}`);
      } finally {
        this.loading = false;
      }
    },

    async viewArtifact(artifact) {
      const isText = (artifact.content_type || '').startsWith('text/');
      this.view = {
        open: true, artifact, content: '', hasMore: false, offset: 0,
        loading: true, isText, rawUrl: null,
      };
      if (!isText) {
        // Binary artifact — get a signed download URL.
        // For images, the UI renders an <img>; for others, a download link.
        try {
          this.view.rawUrl = await MnemoryAPI.getArtifactRawUrl(this.memoryId, artifact.id);
        } catch (err) {
          Alpine.store('notify').error(`Failed to get download URL: ${err.message}`);
          this.view.open = false;
        }
        this.view.loading = false;
        return;
      }
      try {
        const result = await MnemoryAPI.getArtifact(this.memoryId, artifact.id, 0, 5000);
        this.view.content = result.content || '';
        this.view.hasMore = result.has_more || false;
        this.view.offset = (result.content || '').length;
      } catch (err) {
        Alpine.store('notify').error(`Failed to load artifact: ${err.message}`);
        this.view.open = false;
      } finally {
        this.view.loading = false;
      }
    },

    async loadMoreContent() {
      if (!this.view.hasMore || this.view.loading) return;
      this.view.loading = true;
      try {
        const result = await MnemoryAPI.getArtifact(this.memoryId, this.view.artifact.id, this.view.offset, 5000);
        this.view.content += result.content || '';
        this.view.hasMore = result.has_more || false;
        this.view.offset += (result.content || '').length;
      } catch (err) {
        Alpine.store('notify').error(`Failed to load more: ${err.message}`);
      } finally {
        this.view.loading = false;
      }
    },

    /** Check if a content_type is an image type */
    isImage(contentType) {
      return (contentType || '').startsWith('image/');
    },

    async saveArtifact() {
      const f = this.addForm;
      if (!f.content.trim()) {
        Alpine.store('notify').error('Content is required');
        return;
      }
      f.saving = true;
      try {
        const result = await MnemoryAPI.saveArtifact(this.memoryId, {
          content: f.content,
          filename: f.filename || 'note.md',
          content_type: f.content_type || 'text/markdown',
        }, this.revision);
        // result is the artifact metadata
        this.artifacts.push(result);
        f.open = false;
        f.content = '';
        f.filename = 'note.md';
        f.content_type = 'text/markdown';
        Alpine.store('notify').success('Artifact saved');
      } catch (err) {
        if (err.status === 409) {
          window.dispatchEvent(new CustomEvent('mnemory:stale-revision', {
            detail: err.detail || {},
          }));
          this.close();
          Alpine.store('notify').warning('A newer revision exists. The view was refreshed.');
          return;
        }
        Alpine.store('notify').error(`Failed to save artifact: ${err.message}`);
      } finally {
        f.saving = false;
      }
    },

    async deleteArtifact(artifactId) {
      try {
        await MnemoryAPI.deleteArtifact(this.memoryId, artifactId, this.revision);
        this.artifacts = this.artifacts.filter(a => a.id !== artifactId);
        this.deleteConfirm = null;
        // Close view panel if we just deleted the viewed artifact
        if (this.view.artifact?.id === artifactId) {
          this.view.open = false;
        }
        Alpine.store('notify').success('Artifact deleted');
      } catch (err) {
        if (err.status === 409) {
          window.dispatchEvent(new CustomEvent('mnemory:stale-revision', {
            detail: err.detail || {},
          }));
          this.close();
          Alpine.store('notify').warning('A newer revision exists. The view was refreshed.');
          return;
        }
        this.deleteConfirm = null;
        Alpine.store('notify').error(`Failed to delete artifact: ${err.message}`);
      }
    },

    /** Format bytes to human-readable string */
    formatSize(bytes) {
      if (!bytes) return '0 B';
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    },

    /** Format ISO date string for display */
    formatDate(dateStr) {
      if (!dateStr) return '';
      try {
        return new Date(dateStr).toLocaleString(undefined, {
          year: 'numeric', month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit',
        });
      } catch { return dateStr; }
    },
  });

  // ── Session Delete Modal Store ───────────────────────────────
  // Global session delete modal — opened from Sessions tab.
  Alpine.store('sessionDelete', {
    open: false,
    session: null,
    deleteLinkedMemories: false,
    deleting: false,
    onDeleted: null,  // callback after successful deletion

    show(session, onDeleted = null) {
      this.session = session;
      this.deleteLinkedMemories = false;
      this.deleting = false;
      this.onDeleted = onDeleted;
      this.open = true;
    },

    close() {
      this.open = false;
      this.session = null;
      this.deleteLinkedMemories = false;
      this.deleting = false;
      this.onDeleted = null;
    },

    async execute() {
      if (!this.session || this.deleting) return;
      this.deleting = true;
      const sid = this.session.session_id;
      const deleteMems = this.deleteLinkedMemories;
      try {
        const qs = deleteMems ? '?delete_memories=true' : '';
        const result = await MnemoryAPI.del(`/sessions/${sid}${qs}`);
        const msg = deleteMems
          ? `Session deleted (${result.deleted_memories || 0} raw memories removed)`
          : 'Session deleted';
        Alpine.store('notify').success(msg);
        if (this.onDeleted) this.onDeleted(sid);
        this.close();
      } catch (e) {
        Alpine.store('notify').error(`Failed to delete session: ${e.message}`);
        this.deleting = false;
      }
    },

    truncate(str, max = 300) {
      if (!str) return '';
      return str.length > max ? str.substring(0, max) + '...' : str;
    },
  });

  // ── Session Detail Modal Store ────────────────────────────────
  // Global session detail modal — opened from Memories tab when
  // clicking a session_id label.
  Alpine.store('sessionDetail', {
    open: false,
    loading: false,
    session: null,

    async show(sessionId) {
      this.open = true;
      this.loading = true;
      this.session = null;
      try {
        const data = await MnemoryAPI.get(`/sessions/${sessionId}`);
        this.session = data;
      } catch (err) {
        const msg = err.message?.includes('404') || err.message?.includes('not found')
          ? 'Session summary not found — it may predate the session persistence feature'
          : `Failed to load session: ${err.message}`;
        Alpine.store('notify').error(msg);
        this.open = false;
      } finally {
        this.loading = false;
      }
    },

    close() {
      this.open = false;
      this.session = null;
    },

    formatDate(dateStr) {
      if (!dateStr) return '';
      try {
        return new Date(dateStr).toLocaleString(undefined, {
          year: 'numeric', month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit',
        });
      } catch { return dateStr; }
    },

    stateBadgeClass(state) {
      return {
        'bg-yellow-500/20 text-yellow-300': state === 'idle',
        'bg-blue-500/20 text-blue-300': state === 'consolidating',
        'bg-green-500/20 text-green-300': state === 'consolidated',
        'bg-red-500/20 text-red-300': state === 'failed',
      };
    },
  });

  // ── Navigation Store ─────────────────────────────────────────
  Alpine.store('nav', {
    activeTab: 'dashboard',

    setTab(tab) {
      this.activeTab = tab;
      // Dispatch event so tabs can initialize on first view
      window.dispatchEvent(new CustomEvent('mnemory:tab-changed', {
        detail: { tab }
      }));
    },
  });

  // ── Notification Store ───────────────────────────────────────
  Alpine.store('notify', {
    toasts: [],
    _nextId: 0,

    /**
     * Show a toast notification.
     * @param {string} message
     * @param {'success'|'error'|'info'|'warning'} type
     * @param {number} duration - ms before auto-dismiss (0 = manual)
     */
    show(message, type = 'info', duration = 5000) {
      const id = this._nextId++;
      this.toasts.push({ id, message, type, visible: true });
      if (duration > 0) {
        setTimeout(() => this.dismiss(id), duration);
      }
    },

    dismiss(id) {
      const idx = this.toasts.findIndex(t => t.id === id);
      if (idx !== -1) {
        this.toasts[idx].visible = false;
        setTimeout(() => {
          this.toasts = this.toasts.filter(t => t.id !== id);
        }, 300);
      }
    },

    success(msg) { this.show(msg, 'success'); },
    error(msg) { this.show(msg, 'error', 8000); },
    info(msg) { this.show(msg, 'info'); },
    warning(msg) { this.show(msg, 'warning'); },
  });
});
