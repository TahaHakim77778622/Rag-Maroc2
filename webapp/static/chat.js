function toggleUserMenu() {
  const menu = document.getElementById('userDropdown');
  const block = document.getElementById('user-profile-block');
  if (!menu) return;
  const open = menu.style.display === 'none' || menu.style.display === '';
  menu.style.display = open ? 'block' : 'none';
  if (block) block.setAttribute('aria-expanded', open ? 'true' : 'false');
}

document.addEventListener('click', function (e) {
  const wrap = document.querySelector('.sidebar-user-wrap');
  const menu = document.getElementById('userDropdown');
  if (!menu || !wrap) return;
  if (!wrap.contains(e.target)) {
    menu.style.display = 'none';
    const block = document.getElementById('user-profile-block');
    if (block) block.setAttribute('aria-expanded', 'false');
  }
});

(function () {
  const form = document.getElementById('chat-form');
  const messages = document.getElementById('messages');
  const textarea = document.getElementById('q');
  const emptyState = document.getElementById('empty-chat-state');
  const sources = document.getElementById('sources');
  const btn = document.getElementById('send-btn');
  const newChatBtn = document.getElementById('new-chat-btn');
  const fabNewChat = document.getElementById('fab-new-chat');
  const historyPrevBtn = document.getElementById('history-prev-btn');
  const historyNextBtn = document.getElementById('history-next-btn');
  const historyPageInfo = document.getElementById('history-page-info');
  const quickSuggestions = document.getElementById('quick-suggestions');
  const charCounter = document.getElementById('char-counter');
  const hamburgerBtn = document.getElementById('hamburger-btn');
  const sidebar = document.getElementById('sidebar');
  const sidebarBackdrop = document.getElementById('sidebar-backdrop');
  const recentListEl = document.getElementById('recent-conversations');
  const MAX_CHARS = 4000;
  const ACTIVE_CONV_KEY = 'rag_maroc_active_conv_v2';

  let convo = [];
  let conversations = [];
  let currentConversationId = null;
  let historyPage = 1;
  let historyTotalPages = 1;
  const historyPageSize = 40;
  let typingNode = null;

  function getRecentHistoryForRag() {
    return convo.slice(-12);
  }

  function saveActiveConversationId(id) {
    try {
      if (id) localStorage.setItem(ACTIVE_CONV_KEY, String(id));
      else localStorage.removeItem(ACTIVE_CONV_KEY);
    } catch (_e) {}
  }

  function loadActiveConversationId() {
    try {
      var v = localStorage.getItem(ACTIVE_CONV_KEY);
      return v ? parseInt(v, 10) : null;
    } catch (_e) {
      return null;
    }
  }

  function syncEmptyState() {
    if (!emptyState) return;
    var hasMessages = messages.querySelectorAll('.msg:not(.msg-typing)').length > 0;
    emptyState.style.display = hasMessages ? 'none' : '';
  }

  function updateCharCounter() {
    if (!charCounter || !textarea) return;
    var len = textarea.value.length;
    charCounter.textContent = len + ' / ' + MAX_CHARS;
  }

  function closeSidebar() {
    if (!sidebar) return;
    sidebar.classList.remove('is-open');
    if (sidebarBackdrop) sidebarBackdrop.classList.remove('is-visible');
    if (hamburgerBtn) hamburgerBtn.setAttribute('aria-expanded', 'false');
  }

  function openSidebar() {
    if (!sidebar) return;
    sidebar.classList.add('is-open');
    if (sidebarBackdrop) sidebarBackdrop.classList.add('is-visible');
    if (hamburgerBtn) hamburgerBtn.setAttribute('aria-expanded', 'true');
  }

  function toggleSidebar() {
    if (sidebar && sidebar.classList.contains('is-open')) closeSidebar();
    else openSidebar();
  }

  function truncateLabel(s, max) {
    var t = String(s || '').trim();
    max = max || 36;
    if (t.length <= max) return t;
    return t.slice(0, max - 1) + '…';
  }

  async function fetchConversations() {
    try {
      const res = await fetch('/api/chat/conversations', {
        method: 'GET',
        credentials: 'same-origin',
      });
      const data = await res.json();
      conversations = Array.isArray(data.conversations) ? data.conversations : [];
    } catch (_e) {
      conversations = [];
    }
    renderRecentSidebar();
  }

  function renderRecentSidebar() {
    if (!recentListEl) return;
    recentListEl.innerHTML = '';
    if (conversations.length === 0) {
      var empty = document.createElement('li');
      empty.className = 'recent-conversations__empty muted';
      empty.textContent = 'Aucune conversation';
      recentListEl.appendChild(empty);
      return;
    }
    conversations.forEach(function (c) {
      var li = document.createElement('li');
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'recent-conversations__item';
      if (currentConversationId && c.id === currentConversationId) {
        btn.classList.add('is-active');
      }
      var title = c.title || 'Nouvelle conversation';
      var count = parseInt(c.message_count, 10) || 0;
      btn.innerHTML =
        '<span class="recent-conversations__title">' +
        escapeHtml(truncateLabel(title, 40)) +
        '</span>' +
        (count > 0
          ? '<span class="recent-conversations__meta">' + count + ' msg</span>'
          : '<span class="recent-conversations__meta recent-conversations__meta--new">Nouveau</span>');
      btn.title = title;
      btn.addEventListener('click', function () {
        switchConversation(c.id);
      });
      li.appendChild(btn);
      recentListEl.appendChild(li);
    });
  }

  function switchConversation(id) {
    var cid = parseInt(id, 10);
    if (!cid || cid === currentConversationId) {
      closeSidebar();
      return;
    }
    currentConversationId = cid;
    saveActiveConversationId(cid);
    historyPage = 1;
    loadHistory(1);
    renderRecentSidebar();
    closeSidebar();
  }

  async function startNewConversation() {
    try {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        credentials: 'same-origin',
      });
      const data = await res.json();
      currentConversationId = parseInt(data.id, 10) || null;
    } catch (_e) {
      currentConversationId = null;
    }
    saveActiveConversationId(currentConversationId);
    convo = [];
    messages.innerHTML = '';
    historyPage = 1;
    historyTotalPages = 1;
    updatePager();
    sources.innerHTML = '<p class="muted small">Nouvelle conversation — posez votre première question.</p>';
    syncEmptyState();
    await fetchConversations();
    closeSidebar();
    if (textarea) textarea.focus();
  }

  function attachCopyButton(body, text) {
    var actions = document.createElement('div');
    actions.className = 'msg-actions';
    var copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'btn-copy-msg';
    copyBtn.textContent = '📋 Copier';
    copyBtn.addEventListener('click', function () {
      var content = String(text || '');
      function onCopied() {
        copyBtn.textContent = '✓ Copié !';
        window.setTimeout(function () {
          copyBtn.textContent = '📋 Copier';
        }, 2000);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(content).then(onCopied).catch(function () {
          fallbackCopy(content);
          onCopied();
        });
      } else {
        fallbackCopy(content);
        onCopied();
      }
    });
    actions.appendChild(copyBtn);
    body.appendChild(actions);
  }

  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
    } catch (_e) {}
    document.body.removeChild(ta);
  }

  function buildMessageRow(role, text) {
    const wrap = document.createElement('div');
    wrap.className = 'msg msg-' + role;

    const row = document.createElement('div');
    row.className = 'msg-row';

    if (role === 'assistant') {
      const avatar = document.createElement('span');
      avatar.className = 'msg-avatar';
      avatar.setAttribute('aria-hidden', 'true');
      avatar.textContent = 'MA';
      row.appendChild(avatar);
    }

    const body = document.createElement('div');
    body.className = 'msg-body';

    const label = document.createElement('span');
    label.className = 'msg-label';
    label.textContent = role === 'user' ? 'Vous' : 'Assistant';

    const inner = document.createElement('div');
    inner.className = 'msg-inner';
    inner.textContent = text;

    body.appendChild(label);
    body.appendChild(inner);
    if (role === 'assistant') {
      attachCopyButton(body, text);
    }
    row.appendChild(body);
    wrap.appendChild(row);
    return wrap;
  }

  function addBubble(role, text) {
    messages.appendChild(buildMessageRow(role, text));
    messages.scrollTop = messages.scrollHeight;
    syncEmptyState();
  }

  function setTyping(visible) {
    if (visible) {
      if (typingNode) return;
      const wrap = document.createElement('div');
      wrap.className = 'msg msg-assistant msg-typing';
      const row = document.createElement('div');
      row.className = 'msg-row';
      const avatar = document.createElement('span');
      avatar.className = 'msg-avatar';
      avatar.setAttribute('aria-hidden', 'true');
      avatar.textContent = 'MA';
      const body = document.createElement('div');
      body.className = 'msg-body';
      body.innerHTML =
        '<span class="msg-label">Assistant</span>' +
        '<div class="msg-inner typing-indicator" aria-label="Réponse en cours">' +
        '<span></span><span></span><span></span>' +
        '</div>';
      row.appendChild(avatar);
      row.appendChild(body);
      wrap.appendChild(row);
      typingNode = wrap;
      messages.appendChild(wrap);
      messages.scrollTop = messages.scrollHeight;
      syncEmptyState();
      return;
    }
    if (typingNode) {
      typingNode.remove();
      typingNode = null;
      syncEmptyState();
    }
  }

  function autoResizeTextarea() {
    if (!textarea) return;
    textarea.style.height = 'auto';
    var h = Math.min(textarea.scrollHeight, 220);
    textarea.style.height = h + 'px';
    updateCharCounter();
  }

  function renderQuickSuggestions(list) {
    if (!quickSuggestions) return;
    quickSuggestions.innerHTML = '';
    var items = Array.isArray(list) ? list.filter(Boolean).slice(0, 3) : [];
    if (items.length === 0) return;
    items.forEach(function (txt) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip quick-chip';
      b.textContent = String(txt);
      b.addEventListener('click', function () {
        textarea.value = String(txt);
        autoResizeTextarea();
        textarea.focus();
      });
      quickSuggestions.appendChild(b);
    });
  }

  function formatApiError(data) {
    if (!data || typeof data !== 'object') return 'Erreur serveur';
    if (data.error) {
      var err = String(data.error);
      if (data.hint) err += '\n\n' + String(data.hint);
      return err;
    }
    var d = data.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d
        .map(function (x) {
          if (!x) return '';
          if (typeof x === 'string') return x;
          if (x.msg) return String(x.msg);
          return JSON.stringify(x);
        })
        .filter(Boolean)
        .join(' ');
    }
    if (d && typeof d === 'object') return JSON.stringify(d);
    return 'Erreur serveur';
  }

  function updatePager() {
    if (historyPageInfo) {
      historyPageInfo.textContent = 'Page ' + historyPage + ' / ' + historyTotalPages;
    }
    if (historyPrevBtn) historyPrevBtn.disabled = historyPage <= 1;
    if (historyNextBtn) historyNextBtn.disabled = historyPage >= historyTotalPages;
  }

  function scoreBadge(scoreRaw) {
    var sc = parseFloat(scoreRaw);
    if (isNaN(sc)) sc = 0;
    var cls = 'source-score-badge--low';
    var label = 'Faible';
    if (sc > 0.65) {
      cls = 'source-score-badge--high';
      label = 'Fort ✓';
    } else if (sc >= 0.45) {
      cls = 'source-score-badge--medium';
      label = 'Moyen';
    }
    return { cls: cls, label: label, sc: sc };
  }

  function sourceTitleClass(s) {
    var st = String(s.source_type || '').toLowerCase();
    var title = String(s.title || '').toLowerCase();
    if (st === 'web_fallback' || st === 'web') return 'source-title--web';
    if (st === 'bulletin_officiel' || title.indexOf('bulletin') !== -1) return 'source-title--bo';
    var url = String(s.source_url || '');
    if (/^https?:\/\//i.test(url) && st !== 'bulletin_officiel') return 'source-title--web';
    return 'source-title--bo';
  }

  function attachSeeMore(previewEl, fullText) {
    if (!fullText || fullText.length < 180) {
      previewEl.textContent = fullText;
      return;
    }
    previewEl.textContent = fullText;
    previewEl.classList.add('source-preview--clamped');
    var moreBtn = document.createElement('button');
    moreBtn.type = 'button';
    moreBtn.className = 'btn-see-more';
    moreBtn.textContent = 'Voir plus';
    moreBtn.addEventListener('click', function () {
      var expanded = previewEl.classList.toggle('source-preview--expanded');
      previewEl.classList.toggle('source-preview--clamped', !expanded);
      moreBtn.textContent = expanded ? 'Voir moins' : 'Voir plus';
    });
    previewEl.parentNode.appendChild(moreBtn);
  }

  async function loadHistory(page) {
    if (!currentConversationId) {
      convo = [];
      messages.innerHTML = '';
      syncEmptyState();
      return;
    }
    historyPage = Math.max(1, parseInt(page || 1, 10));
    sources.innerHTML = '<p class="muted small">Chargement de la conversation…</p>';
    try {
      const res = await fetch(
        '/api/chat/history?conversation_id=' +
          currentConversationId +
          '&page=' +
          historyPage +
          '&page_size=' +
          historyPageSize,
        { method: 'GET', credentials: 'same-origin' }
      );
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Erreur');
      }
      const hist = Array.isArray(data.history) ? data.history : [];
      historyTotalPages = Math.max(1, parseInt(data.total_pages || 1, 10));
      historyPage = Math.min(historyPage, historyTotalPages);
      updatePager();
      convo = hist
        .filter(function (m) {
          return m && (m.role === 'user' || m.role === 'assistant') && String(m.content || '').trim();
        })
        .map(function (m) {
          return { role: m.role, content: String(m.content) };
        });
      messages.innerHTML = '';
      convo.forEach(function (m) {
        addBubble(m.role, m.content);
      });
      sources.innerHTML = '';
      if (convo.length === 0) {
        sources.innerHTML =
          '<p class="muted small">Conversation vide — posez une question pour commencer.</p>';
      }
    } catch (_err) {
      updatePager();
      sources.innerHTML = '<p class="muted small">Impossible de charger cette conversation.</p>';
    } finally {
      syncEmptyState();
      renderRecentSidebar();
    }
  }

  async function bootstrapChat() {
    await fetchConversations();
    var saved = loadActiveConversationId();
    var found = saved && conversations.some(function (c) {
      return c.id === saved;
    });
    if (found) {
      currentConversationId = saved;
      await loadHistory(1);
      return;
    }
    if (conversations.length > 0) {
      currentConversationId = conversations[0].id;
      saveActiveConversationId(currentConversationId);
      await loadHistory(1);
      return;
    }
    await startNewConversation();
  }

  if (newChatBtn) newChatBtn.addEventListener('click', startNewConversation);
  if (fabNewChat) fabNewChat.addEventListener('click', startNewConversation);

  if (hamburgerBtn) hamburgerBtn.addEventListener('click', toggleSidebar);
  if (sidebarBackdrop) sidebarBackdrop.addEventListener('click', closeSidebar);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = textarea.value.trim();
    const k = parseInt(document.getElementById('k').value, 10) || 5;
    if (!q) return;
    if (!currentConversationId) {
      await startNewConversation();
    }
    addBubble('user', q);
    textarea.value = '';
    autoResizeTextarea();
    btn.disabled = true;
    setTyping(true);
    sources.innerHTML = '<p class="muted small">Recherche…</p>';
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          question: q,
          top_k: k,
          history: getRecentHistoryForRag(),
          conversation_id: currentConversationId,
        }),
      });
      const ct = (res.headers.get('content-type') || '').toLowerCase();
      let data = {};
      try {
        if (ct.indexOf('application/json') !== -1) {
          data = await res.json();
        } else {
          data = { error: (await res.text()) || 'Réponse non JSON' };
        }
      } catch (_parseErr) {
        data = { error: 'Réponse illisible du serveur' };
      }
      if (!res.ok) {
        setTyping(false);
        addBubble('assistant', formatApiError(data));
        sources.innerHTML = '';
        return;
      }
      if (data.conversation_id) {
        currentConversationId = parseInt(data.conversation_id, 10);
        saveActiveConversationId(currentConversationId);
      }
      const ans = data.answer || '';
      setTyping(false);
      addBubble('assistant', ans);
      convo.push({ role: 'user', content: q });
      convo.push({ role: 'assistant', content: ans });
      if (convo.length > 40) convo = convo.slice(-40);
      historyPage = 1;
      sources.innerHTML = '';
      renderQuickSuggestions(data.quick_suggestions);
      await fetchConversations();
      if (data.reliability && typeof data.reliability === 'object') {
        var rel = data.reliability;
        var code = String(rel.code || 'low');
        var relCard = document.createElement('div');
        relCard.className = 'reliability-card reliability-' + escapeHtml(code);
        relCard.innerHTML =
          '<div class="reliability-top">' +
          '<span class="reliability-badge">' +
          escapeHtml(String(rel.label || 'Faible')) +
          '</span>' +
          '<span class="reliability-score">Score: ' +
          escapeHtml(String(rel.score ?? '0')) +
          '</span>' +
          '</div>' +
          '<p class="reliability-detail">' +
          escapeHtml(String(rel.detail || '')) +
          '</p>';
        sources.appendChild(relCard);
      }
      const srcList = Array.isArray(data.sources) ? data.sources : [];
      if (srcList.length === 0) {
        sources.innerHTML =
          '<p class="muted small">Aucun passage dans la réponse JSON. ' +
          'Vérifiez les logs serveur et que <code>vector_store/faiss.index</code> est présent.</p>';
      }
      var feedbackBar = document.createElement('div');
      feedbackBar.className = 'feedback-bar';
      feedbackBar.innerHTML =
        '<span class="muted small">Cette réponse vous a-t-elle aidé ?</span>' +
        '<button type="button" class="btn btn-ghost btn-sm" data-rating="1">Utile</button>' +
        '<button type="button" class="btn btn-ghost btn-sm" data-rating="-1">Pas utile</button>';
      feedbackBar.querySelectorAll('button').forEach(function (fbBtn) {
        fbBtn.addEventListener('click', async function () {
          var rating = parseInt(fbBtn.getAttribute('data-rating'), 10);
          try {
            await fetch('/api/feedback', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              credentials: 'same-origin',
              body: JSON.stringify({ question: q, rating: rating }),
            });
            feedbackBar.innerHTML = '<span class="muted small">Merci pour votre retour.</span>';
          } catch (_e) {
            feedbackBar.innerHTML =
              '<span class="muted small">Impossible d’enregistrer le retour.</span>';
          }
        });
      });
      sources.appendChild(feedbackBar);

      srcList.forEach(function (s, i) {
        const card = document.createElement('div');
        card.className = 'source-card';
        var rawUrl = (s.source_url && String(s.source_url).trim()) || '';
        var safeUrl = /^https?:\/\//i.test(rawUrl) ? rawUrl : '';
        var sourceType = String(s.source_type || '').toLowerCase();
        var originBadge =
          sourceType === 'web_fallback'
            ? '<span class="source-origin-badge">Source web officielle</span>'
            : '';
        var linkHtml = safeUrl
          ? '<a class="source-link" href="' +
            escapeHtml(safeUrl) +
            '" target="_blank" rel="noopener noreferrer">Ouvrir la source ↗</a>'
          : '';
        var badge = scoreBadge(s.score);
        var titleCls = sourceTitleClass(s);
        var previewText = String(s.preview || '');

        card.innerHTML =
          '<div class="source-card__top">' +
          '<div class="source-meta-wrap">' +
          '<span class="source-meta">Source [' +
          (i + 1) +
          ']</span>' +
          originBadge +
          '</div>' +
          '<span class="source-score-badge ' +
          badge.cls +
          '">' +
          escapeHtml(badge.label) +
          ' · ' +
          escapeHtml(badge.sc.toFixed(2)) +
          '</span>' +
          '</div>' +
          '<div class="source-title ' +
          titleCls +
          '">' +
          escapeHtml(s.title || '') +
          '</div>' +
          '<div class="source-sub muted small">' +
          escapeHtml(s.label || '') +
          '</div>' +
          '<div class="source-chunk tiny">' +
          escapeHtml(s.chunk_id || '') +
          '</div>' +
          '<p class="source-explain">' +
          escapeHtml(s.explain || '') +
          '</p>' +
          linkHtml +
          '<div class="source-preview-wrap"><p class="source-preview"></p></div>';

        var previewEl = card.querySelector('.source-preview');
        attachSeeMore(previewEl, previewText);
        sources.appendChild(card);
      });
    } catch (err) {
      setTyping(false);
      addBubble('assistant', 'Erreur réseau : ' + err);
      sources.innerHTML = '';
    } finally {
      setTyping(false);
      btn.disabled = false;
    }
  });

  if (textarea) {
    textarea.addEventListener('input', autoResizeTextarea);
    textarea.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!btn.disabled) form.requestSubmit();
      }
    });
    autoResizeTextarea();
  }

  if (historyPrevBtn) {
    historyPrevBtn.addEventListener('click', function () {
      if (historyPage <= 1) return;
      loadHistory(historyPage - 1);
    });
  }
  if (historyNextBtn) {
    historyNextBtn.addEventListener('click', function () {
      if (historyPage >= historyTotalPages) return;
      loadHistory(historyPage + 1);
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  updatePager();
  bootstrapChat();
  renderQuickSuggestions([]);
})();
