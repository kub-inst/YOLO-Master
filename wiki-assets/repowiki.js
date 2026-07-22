(() => {
  'use strict';
  const pagePrefix = '#/';
  let controller = null;

  window.repoWiki = {
    route(path, anchor = '') {
      return pagePrefix + encodeURIComponent(path) + (anchor ? '?anchor=' + encodeURIComponent(anchor) : '');
    },
    parseRoute() {
      const raw = location.hash;
      if (!raw) return null;
      try {
        if (!raw.startsWith(pagePrefix)) return { path: decodeURIComponent(raw.slice(1)), anchor: '' };
        const [path, query = ''] = raw.slice(pagePrefix.length).split('?');
        return { path: decodeURIComponent(path), anchor: new URLSearchParams(query).get('anchor') || '' };
      } catch (_) { return null; }
    },
    encodedContentUrl(base, path) {
      return base + path.split('/').map(encodeURIComponent).join('/');
    },
    abortPrevious() {
      if (controller) controller.abort();
      controller = new AbortController();
      return controller.signal;
    },
    sanitize(markdown) {
      const template = document.createElement('template');
      template.innerHTML = marked.parse(markdown.replace(/<cite>[\s\S]*?<\/cite>/gi, ''));
      const allowed = new Set(['A','P','DIV','SPAN','H1','H2','H3','H4','H5','H6','UL','OL','LI','PRE','CODE','BLOCKQUOTE','STRONG','EM','DEL','HR','BR','TABLE','THEAD','TBODY','TR','TH','TD','IMG']);
      [...template.content.querySelectorAll('*')].forEach(el => {
        if (!allowed.has(el.tagName)) { el.replaceWith(...el.childNodes); return; }
        [...el.attributes].forEach(attr => {
          const name = attr.name.toLowerCase();
          if (name.startsWith('on') || !['href','src','alt','title','class','id'].includes(name)) el.removeAttribute(attr.name);
        });
        for (const name of ['href', 'src']) {
          const value = el.getAttribute(name);
          if (value && !/^(?:https?:|mailto:|#|\.?\.?\/)/i.test(value)) el.removeAttribute(name);
        }
        if (el.tagName === 'A' && /^https?:/i.test(el.href)) { el.target = '_blank'; el.rel = 'noopener noreferrer'; }
      });
      return template.content;
    },
    wire(body, currentPath, loadPage) {
      body.querySelectorAll('a[href]').forEach(link => link.addEventListener('click', event => {
        const href = link.getAttribute('href');
        if (href.startsWith('#')) {
          event.preventDefault();
          const anchor = decodeURIComponent(href.slice(1));
          history.replaceState(null, '', this.route(currentPath, anchor));
          document.getElementById(anchor)?.scrollIntoView();
        } else if (!/^[a-z]+:/i.test(href) && href.split(/[?#]/)[0].endsWith('.md')) {
          event.preventDefault();
          const target = new URL(href, 'https://wiki.invalid/' + currentPath).pathname.slice(1);
          loadPage(decodeURIComponent(target));
        }
      }));
      body.querySelectorAll('pre code.language-mermaid').forEach(code => {
        const div = document.createElement('div'); div.className = 'mermaid'; div.textContent = code.textContent;
        code.parentElement.replaceWith(div);
      });
      if (window.mermaid) mermaid.run({ nodes: body.querySelectorAll('.mermaid'), suppressErrors: true });
    }
  };
})();
