(function () {
  var tocList = document.getElementById('toc-list');
  if (!tocList) return;

  var content = document.querySelector('.page__content');
  if (!content) return;

  // 收集 h2 和 h3 标题
  var headings = content.querySelectorAll('h2, h3');
  if (headings.length === 0) return;

  // 确保每个标题都有 id
  headings.forEach(function (h, i) {
    if (!h.id) {
      h.id = 'heading-' + i;
    }
  });

  // 生成 TOC 列表
  var fragment = document.createDocumentFragment();
  headings.forEach(function (h) {
    var li = document.createElement('li');
    li.className = h.tagName === 'H3' ? 'toc-h3' : 'toc-h2';
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    li.appendChild(a);
    fragment.appendChild(li);
  });
  tocList.appendChild(fragment);

  // 滚动高亮：IntersectionObserver
  var tocItems = tocList.querySelectorAll('li');

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          // 移除所有 active
          tocItems.forEach(function (item) {
            item.classList.remove('toc-active');
          });
          // 找到对应的 TOC 项并高亮
          var id = entry.target.id;
          var activeLink = tocList.querySelector('a[href="#' + id + '"]');
          if (activeLink) {
            activeLink.parentElement.classList.add('toc-active');
          }
        }
      });
    },
    {
      rootMargin: '-' + 80 + 'px 0px -66% 0px',
      threshold: 0
    }
  );

  headings.forEach(function (h) {
    observer.observe(h);
  });
})();
