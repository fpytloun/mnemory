/**
 * mnemory UI — Memories Browse Tab (Alpine.js component).
 *
 * Lists memories with authorized cursor pagination and revision management.
 * Edit modal and artifact manager are global stores (app.js).
 */

function memoriesTab() {
  return {
    // ── State ──────────────────────────────────────────────────
    memories: [],
    loading: false,
    loadError: '',

    filters: {
      memory_type: '',
      categories: [],
      role: '',
      include_decayed: false,
      labels_json: '',
    },

    sortBy: 'storage',
    filterArtifactsOnly: false,
    filterAgentId: '',
    filterDecayedOnly: false,
    filterMemoryLayer: '',

    pageSize: 50,
    nextCursor: null,
    loadGeneration: 0,
    hasMoreOnServer: false,

    availableCategories: [],
    detailReturnFocus: null,
    detail: {
      open: false,
      memory: null,
      tab: 'current',
      history: null,
      links: null,
      loading: false,
      error: '',
      retractConfirm: false,
      eraseConfirm: false,
      eraseText: '',
    },

    // Add memory modal (local — only triggered from this tab)
    addModal: {
      open: false,
      saving: false,
      showAdvanced: false,
      form: {
        content: '',
        memory_type: '',
        categories: '',
        importance: '',
        pinned: false,
        role: 'user',
        agent_id: '',
        ttl_days: '',
        event_date: '',
        infer: true,
        labels: '',
      },
    },

    deleteConfirm: null,
    initialized: false,

    // Bulk selection
    bulkMode: false,
    selectedIds: [],
    bulkDeleting: false,
    bulkDeleteConfirm: false,

    /** All known agent IDs (loaded from stats API, not from current results) */
    availableAgentIds: [],

    // ── Lifecycle ──────────────────────────────────────────────

    init() {
      window.addEventListener('mnemory:tab-changed', (e) => {
        if (e.detail.tab === 'memories' && !this.initialized) {
          this.initialized = true;
          this.loadMemories(false);
        }
      });

      window.addEventListener('mnemory:user-changed', () => {
        if (this.initialized) {
          this.loadMemories(false);
        }
        this._loadAgentIds();
      });
      window.addEventListener('mnemory:stale-revision', (event) => {
        this.loadMemories(false).then(() => {
          const currentId = event.detail?.current_revision_id;
          const current = this.memories.find((memory) => memory.id === currentId);
          if (current) this.openDetail(current);
        });
      });

      this.loadCategories();
      this._loadAgentIds();
    },

    async loadCategories() {
      try {
        const data = await MnemoryAPI.categories();
        this.availableCategories = (data.categories || []).map((c) => c.name);
      } catch (err) {
        console.warn('Failed to load categories:', err);
      }
    },

    // ── Data Loading ──────────────────────────────────────────

    async loadMemories(append = false) {
      const generation = append ? this.loadGeneration : ++this.loadGeneration;
      this.loadError = '';
      this.loading = true;
      try {
        const params = {
          limit: this.pageSize,
          include_decayed: this.filterDecayedOnly ? true : this.filters.include_decayed,
          decayed_only: this.filterDecayedOnly,
          has_artifacts: this.filterArtifactsOnly,
          memory_layer: this.filterMemoryLayer,
          agent_id: this.filterAgentId,
        };
        if (append && this.nextCursor) params.cursor = this.nextCursor;
        if (this.filters.memory_type) params.memory_type = this.filters.memory_type;
        if (this.filters.categories.length > 0) params.categories = this.filters.categories.join(',');
        if (this.filters.role) params.role = this.filters.role;
        if (this.filters.labels_json) params.labels = this.filters.labels_json;

        const data = await MnemoryAPI.browseMemories(params);
        if (generation !== this.loadGeneration) return;
        const results = data.results || [];
        this.memories = append ? this.memories.concat(results) : results;
        this.nextCursor = data.next_cursor || null;
        this.hasMoreOnServer = !!data.has_more;
      } catch (err) {
        if (generation === this.loadGeneration) {
          this.loadError = err.message;
          Alpine.store('notify').error(`Failed to load memories: ${err.message}`);
        }
      } finally {
        if (generation === this.loadGeneration) this.loading = false;
      }
    },

    loadMore() {
      if (this.hasMoreOnServer && !this.loadError) this.loadMemories(true);
    },

    applyFilters() {
      this.loadMemories(false);
    },

    /** Cursor order is stable point-ID order. */
    onSortChange() {
      this.loadMemories(false);
    },

    // ── Sorting & Filtering ──────────────────────────────────

    /** Load all known agent IDs from the stats API */
    async _loadAgentIds() {
      try {
        const data = await MnemoryAPI.stats();
        this.availableAgentIds = data.agents || [];
      } catch {
        this.availableAgentIds = [];
      }
    },

    get totalFiltered() {
      return this.memories.length;
    },

    /** Paginated slice of filtered+sorted memories */
    get sortedMemories() {
      return this.memories;
    },

    /** Whether the "Load More" button should be visible */
    get canLoadMore() {
      return this.hasMoreOnServer && !this.loadError;
    },

    async openDetail(memory) {
      this.detailReturnFocus = document.activeElement;
      if (!memory.metadata) {
        memory = {
          ...memory,
          _projectionOnly: true,
          metadata: {
            revision: memory.revision,
            revision_state: memory.revision_state,
            memory_type: memory.memory_type,
            memory_layer: memory.memory_layer,
          },
        };
      }
      this.detail = {
        open: true, memory, tab: 'current', history: null, links: null,
        loading: false, error: '', retractConfirm: false,
        eraseConfirm: false, eraseText: '',
      };
      await this.$nextTick();
      this.$refs.memoryDetailClose?.focus();
    },

    closeDetail() {
      this.detail.open = false;
      this.$nextTick(() => this.detailReturnFocus?.focus());
    },

    async selectDetailTab(tab) {
      this.detail.tab = tab;
      if (tab === 'current') return;
      const field = tab === 'history' ? 'history' : 'links';
      if (this.detail[field]) return;
      this.detail.loading = true;
      this.detail.error = '';
      try {
        this.detail[field] = tab === 'history'
          ? await MnemoryAPI.getMemoryHistory(this.detail.memory.id)
          : await MnemoryAPI.getMemoryLinks(this.detail.memory.id);
      } catch (err) {
        this.detail.error = err.message;
      } finally {
        this.detail.loading = false;
      }
    },

    async loadMoreHistory(kind) {
      const history = this.detail.history;
      const cursorKey = kind === 'revision' ? 'next_revision_cursor' : 'next_operation_cursor';
      const cursor = history?.[cursorKey];
      if (!cursor || this.detail.loading) return;
      this.detail.loading = true;
      try {
        const params = kind === 'revision'
          ? { revision_cursor: cursor }
          : { operation_cursor: cursor };
        const page = await MnemoryAPI.getMemoryHistory(this.detail.memory.id, params);
        if (kind === 'revision') history.revisions.unshift(...page.revisions);
        else history.operations.push(...page.operations);
        history[cursorKey] = page[cursorKey];
      } catch (err) {
        this.detail.error = err.message;
      } finally {
        this.detail.loading = false;
      }
    },

    historyEntries() {
      const history = this.detail.history;
      if (!history) return [];
      const revisions = history.revisions.map((value) => ({
        kind: 'revision',
        timestamp: value.metadata?.created_at_utc || '',
        value,
      }));
      const operations = history.operations.map((value) => ({
        kind: 'operation',
        timestamp: value.created_at_utc || '',
        value,
      }));
      return revisions.concat(operations).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    },

    revisionDiff(index) {
      const revisions = this.detail.history?.revisions || [];
      if (index < 1) return [];
      const previous = revisions[index - 1];
      const current = revisions[index];
      const changes = [];
      if (previous.memory !== current.memory) {
        changes.push({ field: 'Text', before: previous.memory, after: current.memory });
      }
      const fields = ['memory_type', 'categories', 'importance', 'pinned', 'ttl_days', 'event_date', 'labels'];
      for (const field of fields) {
        const before = JSON.stringify(previous.metadata?.[field] ?? null);
        const after = JSON.stringify(current.metadata?.[field] ?? null);
        if (before !== after) changes.push({ field, before, after });
      }
      return changes;
    },

    diffForRevision(id) {
      const revisions = this.detail.history?.revisions || [];
      return this.revisionDiff(revisions.findIndex((item) => item.id === id));
    },

    handleDetailTabKey(event) {
      const tabs = ['current', 'history', 'links'];
      const current = tabs.indexOf(this.detail.tab);
      const delta = event.key === 'ArrowRight' ? 1 : -1;
      const next = tabs[(current + delta + tabs.length) % tabs.length];
      this.selectDetailTab(next);
      this.$nextTick(() => document.getElementById(`memory-tab-${next}`)?.focus());
    },

    trapDetailFocus(event) {
      const dialog = event.currentTarget;
      const focusable = [...dialog.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), [tabindex="0"]'
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
    },

    // ── Add Memory ────────────────────────────────────────────

    openAdd() {
      this.addModal = {
        open: true,
        saving: false,
        showAdvanced: false,
        form: {
          content: '',
          memory_type: '',
          categories: '',
          importance: '',
          pinned: false,
          role: 'user',
          agent_id: '',
          ttl_days: '',
          event_date: '',
          infer: true,
          labels: '',
        },
      };
    },

    async saveAdd() {
      const f = this.addModal.form;
      if (!f.content.trim()) {
        Alpine.store('notify').error('Content is required');
        return;
      }
      this.addModal.saving = true;
      try {
        const payload = { content: f.content, infer: f.infer, role: f.role };
        if (f.memory_type) payload.memory_type = f.memory_type;
        if (f.importance) payload.importance = f.importance;
        if (f.pinned) payload.pinned = true;
        if (f.agent_id.trim()) payload.agent_id = f.agent_id.trim();
        if (f.ttl_days !== '') {
          const ttl = parseInt(f.ttl_days, 10);
          if (!isNaN(ttl)) payload.ttl_days = ttl;
        }
        if (f.event_date.trim()) payload.event_date = f.event_date.trim();
        const cats = f.categories ? f.categories.split(',').map(c => c.trim()).filter(Boolean) : [];
        if (cats.length > 0) payload.categories = cats;
        if (f.labels) {
          try {
            const labels = JSON.parse(f.labels);
            if (Object.keys(labels).length > 0) payload.labels = labels;
          } catch (e) {
            // Invalid JSON — skip labels
          }
        }

        const result = await MnemoryAPI.addMemory(payload);
        // result.results is an array of {id, memory, event}
        const added = (result.results || []);
        if (added.length > 0) {
          // Reload to get full metadata; or prepend a minimal item
          await this.loadMemories(false);
          Alpine.store('notify').success(`Memory added (${added.length} fact${added.length > 1 ? 's' : ''} stored)`);
        } else {
          Alpine.store('notify').info('No new facts extracted — memory may already exist');
        }
        this.addModal.open = false;
      } catch (err) {
        Alpine.store('notify').error(`Failed to add memory: ${err.message}`);
      } finally {
        this.addModal.saving = false;
      }
    },

    // ── Edit Memory (delegates to global store) ───────────────

    openEdit(mem) {
      const returnFocus = this.detailReturnFocus;
      this.detail.open = false;
      Alpine.store('memoryEdit').show(mem, () => {
        this.loadMemories(false);
      }, returnFocus);
    },

    // ── Artifacts (delegates to global store) ─────────────────

    openArtifacts(mem) {
      const returnFocus = this.detailReturnFocus;
      this.detail.open = false;
      Alpine.store('artifactMgr').show(mem, returnFocus);
    },

    // ── Bulk Selection ─────────────────────────────────────────

    toggleBulkMode() {
      this.bulkMode = !this.bulkMode;
      if (!this.bulkMode) {
        this.selectedIds = [];
        this.bulkDeleteConfirm = false;
      }
    },

    toggleSelect(id) {
      const idx = this.selectedIds.indexOf(id);
      if (idx === -1) {
        this.selectedIds.push(id);
      } else {
        this.selectedIds.splice(idx, 1);
      }
    },

    isSelected(id) {
      return this.selectedIds.includes(id);
    },

    /** Select all currently visible (filtered+paginated) memories */
    selectAll() {
      this.selectedIds = this.sortedMemories.map(m => m.id);
    },

    deselectAll() {
      this.selectedIds = [];
    },

    get allSelected() {
      return this.sortedMemories.length > 0 &&
        this.sortedMemories.every(m => this.selectedIds.includes(m.id));
    },

    get selectedCount() {
      return this.selectedIds.length;
    },

    async bulkDelete() {
      if (this.selectedIds.length === 0) return;
      this.bulkDeleting = true;
      const toDelete = [...this.selectedIds];
      let deleted = 0;
      let failed = 0;

      // Fire all deletes in parallel (batches of 10)
      for (let i = 0; i < toDelete.length; i += 10) {
        const batch = toDelete.slice(i, i + 10);
        const results = await Promise.allSettled(
          batch.map(id => {
            const memory = this.memories.find((item) => item.id === id);
            return MnemoryAPI.deleteMemory(id, memory?.metadata?.revision);
          })
        );
        for (let j = 0; j < results.length; j++) {
          if (results[j].status === 'fulfilled') {
            deleted++;
          } else {
            failed++;
          }
        }
      }

      await this.loadMemories(false);

      this.selectedIds = [];
      this.bulkDeleteConfirm = false;
      this.bulkDeleting = false;
      this.bulkMode = false;

      if (failed > 0) {
        Alpine.store('notify').warning(`Retracted ${deleted} memories, ${failed} failed`);
      } else {
        Alpine.store('notify').success(`Retracted ${deleted} memories`);
      }
    },

    // ── Delete ────────────────────────────────────────────────

    async deleteMemory(id) {
      const memory = this.memories.find((item) => item.id === id) || this.detail.memory;
      try {
        await MnemoryAPI.deleteMemory(id, memory?.metadata?.revision);
        this.memories = this.memories.filter((m) => m.id !== id);
        this.deleteConfirm = null;
        this.closeDetail();
        Alpine.store('notify').success('Memory retracted. Its history remains available.');
      } catch (err) {
        if (err.status === 409) {
          window.dispatchEvent(new CustomEvent('mnemory:stale-revision', {
            detail: err.detail || {},
          }));
          Alpine.store('notify').warning('A newer revision exists. The view was refreshed.');
          return;
        }
        this.deleteConfirm = null;
        Alpine.store('notify').error(`Failed to retract: ${err.message}`);
      }
    },

    async privacyErase() {
      if (this.detail.eraseText !== 'ERASE') return;
      const id = this.detail.memory.id;
      try {
        await MnemoryAPI.privacyEraseMemory(id);
        this.memories = this.memories.filter((memory) => memory.id !== id);
        this.closeDetail();
        Alpine.store('notify').success('The memory lineage was permanently erased.');
      } catch (err) {
        Alpine.store('notify').error(`Privacy erase failed: ${err.message}`);
      }
    },

    // ── Clipboard ─────────────────────────────────────────────

    copyId(id) {
      navigator.clipboard.writeText(id).then(
        () => Alpine.store('notify').success('ID copied to clipboard'),
        () => Alpine.store('notify').error('Failed to copy ID'),
      );
    },

    // ── Display Helpers ───────────────────────────────────────

    typeBadgeClass(type) {
      const classes = {
        fact: 'badge-fact',
        preference: 'badge-preference',
        episodic: 'badge-episodic',
        procedural: 'badge-procedural',
        context: 'badge-context',
      };
      return classes[type] || 'badge-context';
    },

    importanceBadgeClass(importance) {
      const classes = {
        low: 'badge-low',
        normal: 'badge-normal',
        high: 'badge-high',
        critical: 'badge-critical',
      };
      return classes[importance] || 'badge-normal';
    },

    truncate(str, max = 120) {
      if (!str) return '';
      return str.length > max ? str.substring(0, max) + '...' : str;
    },

    formatDate(dateStr) {
      if (!dateStr) return '';
      try {
        const d = new Date(dateStr);
        return d.toLocaleString(undefined, {
          year: 'numeric', month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit',
        });
      } catch { return dateStr; }
    },
  };
}
