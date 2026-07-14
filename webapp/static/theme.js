(() => {
  const btn = document.getElementById('theme-toggle');
  const root = document.documentElement;
  const metaTheme = document.querySelector('meta[name="theme-color"]');

  function currentTheme() {
    const t = root.getAttribute('data-theme');
    return t === 'light' ? 'light' : 'dark';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('theme', theme);
    } catch (_e) {}
    if (metaTheme) {
      metaTheme.setAttribute('content', theme === 'light' ? '#f3f7fc' : '#0c1017');
    }
    if (btn) {
      const next = theme === 'light' ? 'sombre' : 'clair';
      btn.textContent = theme === 'light' ? '🌙' : '☀️';
      btn.title = `Passer en mode ${next}`;
      btn.setAttribute('aria-label', `Passer en mode ${next}`);
    }
  }

  const initial = currentTheme();
  applyTheme(initial);

  if (!btn) return;
  btn.addEventListener('click', () => {
    const next = currentTheme() === 'light' ? 'dark' : 'light';
    applyTheme(next);
  });
})();
