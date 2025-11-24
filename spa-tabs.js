// SPA tab behavior: intercept navbar clicks and show/hide main sections
document.addEventListener('DOMContentLoaded', () => {
  const navLinks = document.querySelectorAll('.nav-links a');
  const sections = {
    dashboard: document.getElementById('dashboard'),
    analysis: document.getElementById('analysis'),
    predictions: document.getElementById('predictions'),
    'market-news': document.getElementById('market-news')
  };

  function hideAll() {
    Object.values(sections).forEach(el => {
      if (!el) return;
      el.classList.add('hidden');
    });
  }

  function setActiveNav(hash) {
    navLinks.forEach(n => n.classList.remove('active'));
    const link = Array.from(navLinks).find(n => n.getAttribute('href') === `#${hash}`);
    if (link) link.classList.add('active');
  }

  function showSection(name, replaceHash = true) {
    if (!name) name = 'dashboard';

    if (!(name in sections)) name = 'dashboard';

    hideAll();
    const el = sections[name];
    if (el) {
      el.classList.remove('hidden');
      // scroll to top of the revealed section
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // If we should update the URL hash, use history API to avoid duplicate history entries
    if (replaceHash) {
      history.replaceState(null, '', `#${name}`);
    }

    setActiveNav(name);

    // If navigating to predictions, highlight it
    if (name === 'predictions' && el) {
      el.classList.add('highlight');
      setTimeout(() => el.classList.remove('highlight'), 1500);
    }

    // Load data when switching to analysis tab
    if (name === 'analysis' && typeof window.loadAnalysisData === 'function') {
      window.loadAnalysisData();
    }

    // Load data when switching to market-news tab
    if (name === 'market-news' && typeof window.loadMarketSentimentData === 'function') {
      window.loadMarketSentimentData();
    }
  }

  // Attach click handlers
  navLinks.forEach(a => {
    a.addEventListener('click', (e) => {
      const href = a.getAttribute('href') || '';
      if (!href.startsWith('#')) return; // allow external
      e.preventDefault();
      const target = href.slice(1);
      // push a new hash so back/forward works
      history.pushState(null, '', `#${target}`);
      showSection(target, false);
    });
  });

  // Handle back/forward and direct-links
  window.addEventListener('popstate', () => {
    const hash = location.hash ? location.hash.slice(1) : 'dashboard';
    showSection(hash, false);
  });

  // Initial load: show section based on hash
  const initial = location.hash ? location.hash.slice(1) : 'dashboard';
  showSection(initial, false);
});
