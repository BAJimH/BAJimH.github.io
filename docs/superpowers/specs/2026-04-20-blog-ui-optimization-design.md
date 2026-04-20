# Blog UI 优化设计文档

**日期**: 2026-04-20
**项目**: BAJimH.github.io (Jekyll + Academic Pages 主题)

## 目标

在保留现有 Academic Pages / Minimal Mistakes 主题视觉风格的前提下，完成三项功能增强：

1. 侧边栏按页面类型显示/隐藏
2. 博客文章页增加左侧 TOC 导航
3. 接入不蒜子阅读量统计

## 约束

- **不改动**：配色（`$primary-color: #7a8288`, `$info-color: #52adc8`, `$body-color: #fff` 等）、字体栈、字号体系（`$type-size-1` ~ `$type-size-8`）、间距、导航栏样式、代码块样式
- **不引入**：新的 CSS 框架、渐变色、卡片化设计、圆角阴影等与现有风格不符的视觉元素
- **兼容**：保持 GitHub Pages 可部署、移动端响应式

## 设计细节

### 1. 侧边栏显示控制

**当前行为**: 所有页面在 front matter 中设置 `author_profile: true`，统一显示左侧个人信息侧边栏。

**目标行为**:
- 首页（`permalink: /`）：显示侧边栏（保持不变）
- 博客文章页（`_posts/*`）：隐藏侧边栏，改为 TOC（见下节）
- 其他页面（Publications, CV, archive 等）：隐藏侧边栏，内容区域扩展

**实现方式**:
- 在各页面的 front matter 或 `_config.yml` 的 defaults 中设置 `author_profile: false`
- 仅首页 `_pages/about.md` 保留 `author_profile: true`
- 对于隐藏侧边栏的非博客页面，通过 CSS class 将内容区域从 `span(10 of 12)` 扩展到 `span(12 of 12)`，移除 prefix

### 2. 博客文章页左侧 TOC

**位置**: 复用原侧边栏位置（左侧 `span(2 of 12)` 区域）

**行为**:
- 自动提取文章中的 `h2` / `h3` 标题生成目录
- `position: sticky` 固定定位，随页面滚动保持可见
- 当前阅读章节高亮，使用现有链接色 `$info-color: #52adc8`
- 移动端（`< $large` 断点）TOC 隐藏（`display: none`），不显示目录

**实现方式**:
- 新建 `_includes/toc-sidebar.html`，通过 JS 动态扫描文章中的 h2/h3 生成目录（比 Jekyll `{:toc}` 更灵活，支持滚动高亮联动）
- 新建 `_layouts/post.html`（继承 `default.html`），用 TOC 替代 author-profile 侧边栏
- 在 `_sass/_sidebar.scss` 中添加 `.toc-sidebar` 样式，复用 `.sidebar` 的宽度和定位逻辑
- 滚动高亮通过轻量 JS（IntersectionObserver）实现

**样式规范**:
- 目录标题：`$type-size-6`（0.75em），颜色 `$dark-gray`
- 目录项：`$type-size-6`，颜色 `$gray`（`#7a8288`）
- 当前项：颜色 `$info-color`（`#52adc8`），font-weight 600
- 行高：1.5，与原侧边栏一致

### 3. 不蒜子阅读量统计

**接入方式**: 在 `_includes/` 中引入不蒜子 script，零配置。

**显示位置**:
- 博客文章页：在文章 meta 信息行（日期 / 阅读时间旁）显示单篇阅读量（`busuanzi_value_page_pv`）
- 可选：页脚显示全站总访问量（`busuanzi_value_site_pv`）

**实现方式**:
- 在 `_layouts/post.html` 的 header 区域中，`read_time` 旁追加 `<span id="busuanzi_value_page_pv"></span>`
- 在 `_includes/footer.html` 或 `_layouts/default.html` 的 `<head>` 中添加：
  `<script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>`
- 样式沿用现有 `.page__meta` 的字号和颜色

## 需修改的文件清单

| 文件 | 改动类型 | 描述 |
|------|---------|------|
| `_config.yml` | 修改 | defaults 中 posts 的 `author_profile` 改为 false，添加 `toc_sidebar: true` |
| `_pages/about.md` | 保持 | `author_profile: true`（已有） |
| `_pages/*.md/html` (非首页) | 修改 | `author_profile: false` |
| `_layouts/post.html` | 新建 | 博客文章专用 layout，含 TOC 侧边栏 + 不蒜子 |
| `_includes/toc-sidebar.html` | 新建 | TOC 侧边栏组件 |
| `_sass/_sidebar.scss` | 修改 | 添加 `.toc-sidebar` 样式 |
| `_sass/_page.scss` | 修改 | 无侧边栏页面的内容区域扩展样式 |
| `_layouts/default.html` | 修改 | `<head>` 中引入不蒜子 script |
| `assets/js/toc-highlight.js` | 新建 | TOC 滚动高亮（IntersectionObserver） |

## 不做的事情

- 不修改配色方案、字体、字号
- 不引入新的 UI 框架或设计系统
- 不重新设计导航栏或页脚
- 不修改文章内容的排版样式
- 不添加暗色模式
