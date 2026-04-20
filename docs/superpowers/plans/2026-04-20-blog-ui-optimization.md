# Blog UI 优化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留现有 Academic Pages 主题风格的前提下，实现侧边栏按页面类型显示/隐藏、博客文章页左侧 TOC、不蒜子阅读量统计。

**Architecture:** 通过 `_config.yml` defaults 控制 `author_profile` 的页面级开关；新建 `_layouts/post.html` 作为博客文章专用 layout，用 TOC 侧边栏替代 author-profile；不蒜子 script 在 `_includes/head/custom.html` 中全局引入，阅读量在 post layout 的 header 中显示。

**Tech Stack:** Jekyll, Liquid, SCSS (Susy grid), Vanilla JS (IntersectionObserver)

---

## 文件结构

| 文件 | 职责 | 操作 |
|------|------|------|
| `_config.yml` | 站点配置，defaults 中控制各类页面的 `author_profile` 和 `layout` | 修改 |
| `_pages/about.md` | 首页，保留 `author_profile: true` | 不动 |
| `_pages/cv.md` | CV 页 | 修改 front matter |
| `_pages/publications.html` | 发表页 | 修改 front matter |
| `_pages/year-archive.html` | 博客归档页 | 修改 front matter |
| `_pages/category-archive.html` | 分类归档页 | 修改 front matter |
| `_pages/tag-archive.html` | 标签归档页 | 修改 front matter |
| `_layouts/post.html` | 博客文章专用 layout：含 TOC + 不蒜子阅读量 | 新建 |
| `_includes/toc-sidebar.html` | TOC 侧边栏 HTML 容器 | 新建 |
| `assets/js/toc-highlight.js` | TOC 生成 + 滚动高亮 JS | 新建 |
| `_sass/_sidebar.scss` | 添加 `.toc-sidebar` 样式 | 修改 |
| `_sass/_page.scss` | 添加 `.page--wide` 样式（无侧边栏页面内容扩展） | 修改 |
| `_sass/_archive.scss` | 添加 `.archive--wide` 样式（archive 布局的全宽版本） | 修改 |
| `_layouts/archive.html` | 条件添加 `archive--wide` class | 修改 |
| `_layouts/single.html` | 条件添加 `page--wide` class | 修改 |
| `_includes/head/custom.html` | 引入不蒜子 script | 修改 |

---

### Task 1: 配置 `_config.yml` — 全局关闭 author_profile，posts 使用 post layout

**Files:**
- Modify: `_config.yml:237-291`

- [ ] **Step 1: 修改 `_config.yml` defaults 中 posts 的配置**

将 `_posts` 的 `layout` 改为 `post`（新 layout），`author_profile` 改为 `false`：

```yaml
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: post
      author_profile: false
      read_time: true
      comments: true
      share: true
      related: true
```

- [ ] **Step 2: 修改 `_config.yml` defaults 中 pages 的配置**

将 `_pages` 的 `author_profile` 改为 `false`：

```yaml
  # _pages
  - scope:
      path: ""
      type: pages
    values:
      layout: single
      author_profile: false
```

- [ ] **Step 3: 修改 `_config.yml` defaults 中 publications 的配置**

```yaml
  # _publications
  - scope:
      path: ""
      type: publications
    values:
      layout: single
      author_profile: false
      share: true
      comments: true
```

- [ ] **Step 4: 确保首页 `_pages/about.md` 保留 `author_profile: true`**

`about.md` 的 front matter 已有 `author_profile: true`，无需修改。确认内容：

```yaml
---
permalink: /
title: "Zhaojun Huang's Personal Site"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
```

- [ ] **Step 5: Commit**

```bash
git add _config.yml
git commit -m "feat: disable author_profile globally, enable only on homepage"
```

---

### Task 2: 修改各 `_pages` 文件 — 确保非首页的 `author_profile` 为 false

**Files:**
- Modify: `_pages/cv.md` (front matter)
- Modify: `_pages/publications.html` (front matter)
- Modify: `_pages/year-archive.html` (front matter)
- Modify: `_pages/category-archive.html` (front matter)
- Modify: `_pages/tag-archive.html` (front matter)

注意：`_config.yml` defaults 已将 pages 默认设为 `author_profile: false`，但部分页面 front matter 中有显式的 `author_profile: true` 会覆盖 defaults。需要删除或改为 false。

- [ ] **Step 1: 修改 `_pages/cv.md`**

将 front matter 中的 `author_profile: true` 删除（让 defaults 生效）：

```yaml
---
layout: archive
title: "CV"
permalink: /cv/
redirect_from:
  - /resume
---
```

- [ ] **Step 2: 修改 `_pages/publications.html`**

删除 `author_profile: true`：

```yaml
---
layout: archive
title: "Publications"
permalink: /publications/
---
```

- [ ] **Step 3: 修改 `_pages/year-archive.html`**

删除 `author_profile: true`：

```yaml
---
layout: archive
permalink: /year-archive/
title: "Blog posts"
redirect_from:
  - /wordpress/blog-posts/
---
```

- [ ] **Step 4: 修改 `_pages/category-archive.html`**

删除 `author_profile: true`：

```yaml
---
layout: archive
permalink: /categories/
title: "Posts by Category"
---
```

- [ ] **Step 5: 修改 `_pages/tag-archive.html`**

删除 `author_profile: true`：

```yaml
---
layout: archive
permalink: /tags/
title: "Posts by Tags"
---
```

- [ ] **Step 6: Commit**

```bash
git add _pages/cv.md _pages/publications.html _pages/year-archive.html _pages/category-archive.html _pages/tag-archive.html
git commit -m "feat: remove author_profile from non-homepage pages"
```

---

### Task 3: 添加宽布局 CSS 样式 — 无侧边栏页面的内容区域扩展

**Files:**
- Modify: `_sass/_page.scss:19-35`
- Modify: `_sass/_archive.scss:5-16`
- Modify: `_layouts/single.html:20`
- Modify: `_layouts/archive.html:18`

当没有侧边栏时，`.page` 和 `.archive` 的内容需要从 `span(10 of 12)` 扩展。sidebar.html 中通过 `{% if page.author_profile %}` 控制渲染，但 CSS grid 仍然预留了侧边栏空间。

- [ ] **Step 1: 在 `_sass/_page.scss` 中，在 `.page` 规则块（第 35 行）之后添加 `.page--wide`**

```scss
.page--wide {
  @include breakpoint($large) {
    @include span(12 of 12);
    @include prefix(0);
    @include suffix(0);
  }
}
```

- [ ] **Step 2: 在 `_sass/_archive.scss` 中，在 `.archive` 规则块（第 28 行 `}` 之后）添加 `.archive--wide`**

```scss
.archive--wide {
  @include breakpoint($large) {
    @include span(12 of 12);
    @include prefix(0);
  }
}
```

- [ ] **Step 3: 修改 `_layouts/single.html` 第 20 行 — 条件添加 `page--wide` class**

将：
```html
  <article class="page" itemscope itemtype="http://schema.org/CreativeWork">
```

改为：
```html
  <article class="page{% unless page.author_profile or layout.author_profile %} page--wide{% endunless %}" itemscope itemtype="http://schema.org/CreativeWork">
```

- [ ] **Step 4: 修改 `_layouts/archive.html` 第 18 行 — 条件添加 `archive--wide` class**

将：
```html
  <div class="archive">
```

改为：
```html
  <div class="archive{% unless page.author_profile or layout.author_profile %} archive--wide{% endunless %}">
```

- [ ] **Step 5: Commit**

```bash
git add _sass/_page.scss _sass/_archive.scss _layouts/single.html _layouts/archive.html
git commit -m "feat: add wide layout classes for full-width content when no sidebar"
```

---

### Task 4: 添加 TOC 侧边栏样式

**Files:**
- Modify: `_sass/_sidebar.scss` (在文件末尾追加)

- [ ] **Step 1: 在 `_sass/_sidebar.scss` 文件末尾追加 `.toc-sidebar` 样式**

```scss
/*
   TOC Sidebar (left side, replaces author profile on post pages)
   ========================================================================== */

.toc-sidebar {
  display: none;

  @include breakpoint($large) {
    display: block;
    position: sticky;
    top: $masthead-height + 1em;
    max-height: calc(100vh - #{$masthead-height} - 2em);
    overflow-y: auto;
  }

  .toc-title {
    font-family: $sans-serif-narrow;
    font-size: $type-size-6;
    font-weight: bold;
    color: $dark-gray;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5em;
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  li {
    font-family: $sans-serif;
    font-size: $type-size-6;
    line-height: 1.5;
    margin-bottom: 0.4em;
  }

  li a {
    color: $gray;
    text-decoration: none;

    &:hover {
      color: $info-color;
    }
  }

  li.toc-active > a {
    color: $info-color;
    font-weight: 600;
  }

  li.toc-h3 {
    padding-left: 1em;
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add _sass/_sidebar.scss
git commit -m "feat: add toc-sidebar styles for post TOC navigation"
```

---

### Task 5: 创建 TOC 侧边栏 HTML 组件

**Files:**
- Create: `_includes/toc-sidebar.html`

- [ ] **Step 1: 创建 `_includes/toc-sidebar.html`**

```html
<nav class="toc-sidebar">
  <p class="toc-title">{{ site.data.ui-text[site.locale].toc_label | default: "目录" }}</p>
  <ul id="toc-list">
    <!-- JS 动态填充 -->
  </ul>
</nav>
```

- [ ] **Step 2: Commit**

```bash
git add _includes/toc-sidebar.html
git commit -m "feat: add toc-sidebar HTML component"
```

---

### Task 6: 创建 TOC 生成 + 滚动高亮 JS

**Files:**
- Create: `assets/js/toc-highlight.js`

- [ ] **Step 1: 创建 `assets/js/toc-highlight.js`**

```javascript
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
```

- [ ] **Step 2: Commit**

```bash
git add assets/js/toc-highlight.js
git commit -m "feat: add TOC generation and scroll-highlight JS"
```

---

### Task 7: 创建博客文章专用 layout `_layouts/post.html`

**Files:**
- Create: `_layouts/post.html`

这是核心文件，基于现有 `_layouts/single.html` 修改：用 TOC 替代 sidebar，在 header 中添加不蒜子阅读量。

- [ ] **Step 1: 创建 `_layouts/post.html`**

基于 `_layouts/single.html` 的结构，做以下改动：
1. 用 `{% include toc-sidebar.html %}` 替代 `{% include sidebar.html %}`
2. 在 header 中 read_time 旁添加不蒜子阅读量
3. 文件末尾引入 `toc-highlight.js`
4. `.page` 不添加 `page--wide`（因为有 TOC 侧边栏占位）

```html
---
layout: default
---

{% include base_path %}

{% if page.header.overlay_color or page.header.overlay_image or page.header.image %}
  {% include page__hero.html %}
{% endif %}

{% if page.url != "/" and site.breadcrumbs %}
  {% unless paginator %}
    {% include breadcrumbs.html %}
  {% endunless %}
{% endif %}

<div id="main" role="main">
  <div class="sidebar sticky">
    {% include toc-sidebar.html %}
  </div>

  <article class="page" itemscope itemtype="http://schema.org/CreativeWork">
    {% if page.title %}<meta itemprop="headline" content="{{ page.title | markdownify | strip_html | strip_newlines | escape_once }}">{% endif %}
    {% if page.excerpt %}<meta itemprop="description" content="{{ page.excerpt | markdownify | strip_html | strip_newlines | escape_once }}">{% endif %}
    {% if page.date %}<meta itemprop="datePublished" content="{{ page.date | date: "%B %d, %Y" }}">{% endif %}
    {% if page.modified %}<meta itemprop="dateModified" content="{{ page.modified | date: "%B %d, %Y" }}">{% endif %}

    <div class="page__inner-wrap">
      {% unless page.header.overlay_color or page.header.overlay_image %}
        <header>
          {% if page.title %}<h1 class="page__title" itemprop="headline">{{ page.title | markdownify | remove: "<p>" | remove: "</p>" }}</h1>{% endif %}
          <p class="page__meta">
            {% if page.date %}
              <i class="fa fa-fw fa-calendar" aria-hidden="true"></i> <time datetime="{{ page.date | date_to_xmlschema }}">{{ page.date | default: "1900-01-01" | date: "%B %d, %Y" }}</time>
            {% endif %}
            {% if page.read_time %}
              &nbsp;&middot;&nbsp; <i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}
            {% endif %}
            &nbsp;&middot;&nbsp; <i class="fa fa-eye" aria-hidden="true"></i>
            <span id="busuanzi_value_page_pv">-</span> 次阅读
          </p>
        </header>
      {% endunless %}

      <section class="page__content" itemprop="text">
        {{ content }}

        {% if page.link %}<div><a href="{{ page.link }}" class="btn">{{ site.data.ui-text[site.locale].ext_link_label | default: "Direct Link" }}</a></div>{% endif %}
      </section>

      <footer class="page__meta">
        {% if site.data.ui-text[site.locale].meta_label %}
          <h4 class="page__meta-title">{{ site.data.ui-text[site.locale].meta_label }}</h4>
        {% endif %}
        {% include page__taxonomy.html %}
      </footer>

      {% if page.share %}{% include social-share.html %}{% endif %}

      {% include post_pagination.html %}
    </div>

    {% if site.comments.provider and page.comments %}
      {% include comments.html %}
    {% endif %}
  </article>

  {% comment %}<!-- only show related on a post page when not disabled -->{% endcomment %}
  {% if page.id and page.related and site.related_posts.size > 0 %}
    <div class="page__related">
      {% if site.data.ui-text[site.locale].related_label %}
        <h4 class="page__related-title">{{ site.data.ui-text[site.locale].related_label | default: "You May Also Enjoy" }}</h4>
      {% endif %}
      <div class="grid__wrapper">
        {% for post in site.related_posts limit:4 %}
          {% include archive-single.html type="grid" %}
        {% endfor %}
      </div>
    </div>
  {% endif %}
</div>

<script src="{{ base_path }}/assets/js/toc-highlight.js"></script>
```

- [ ] **Step 2: Commit**

```bash
git add _layouts/post.html
git commit -m "feat: add post layout with TOC sidebar and busuanzi view count"
```

---

### Task 8: 引入不蒜子 script 到全局 head

**Files:**
- Modify: `_includes/head/custom.html`

- [ ] **Step 1: 在 `_includes/head/custom.html` 的 `<!-- end custom head snippets -->` 前添加不蒜子 script**

在文件最末尾 `<!-- end custom head snippets -->` 之前添加：

```html
<!-- 不蒜子阅读量统计 -->
<script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>
```

- [ ] **Step 2: 移除旧的 `page__views.html` 中的不蒜子引用（避免重复加载）**

由于 `_includes/page__views.html` 中也有不蒜子 script 的引用（在 LeanCloud 未启用时作为 fallback），而现在我们在全局 head 中已引入，需要清理旧代码。

修改 `_layouts/post.html` — 确认不再 `{% include page__views.html %}`（新 layout 中已经不包含它，所以无需操作）。

而 `_layouts/single.html` 中仍有 `{% include page__views.html %}`（第 66-67 行），由于 single.html 现在只用于非 post 页面，且 `page__views.html` 依赖 `page.id`（只有 posts 有），所以不会触发，无需修改。

- [ ] **Step 3: Commit**

```bash
git add _includes/head/custom.html
git commit -m "feat: add busuanzi script to global head for page view counting"
```

---

### Task 9: 本地验证

**Files:** 无新增修改

- [ ] **Step 1: 启动 Jekyll 本地服务器**

```bash
cd /mnt/user-ssd/huangzhaojun/blogs/BAJimH.github.io
bundle exec jekyll serve -l -H localhost
```

如果 `bundle` 未安装，用：
```bash
gem install jekyll bundler && bundle install && bundle exec jekyll serve -l -H localhost
```

- [ ] **Step 2: 验证首页**

打开 `http://localhost:4000/`，确认：
- 左侧个人信息侧边栏正常显示
- 内容区域布局不变

- [ ] **Step 3: 验证博客文章页**

点击任一博客文章，确认：
- 左侧无个人信息，显示 TOC 目录
- TOC 正确提取了文章中的 h2/h3
- 滚动时 TOC 当前章节高亮
- 文章标题下方显示日期、阅读时间、阅读量
- TOC 使用 sticky 定位

- [ ] **Step 4: 验证 Publications / CV 等页面**

打开 `/publications/`、`/cv/` 等页面，确认：
- 无侧边栏
- 内容区域为全宽（`page--wide`）

- [ ] **Step 5: 验证移动端响应式**

使用浏览器开发者工具切换到移动端视图（< 925px），确认：
- 首页：侧边栏折叠为内联显示（原有行为）
- 博客文章页：TOC 不显示（`display: none`）
- 其他页面：正常全宽显示
