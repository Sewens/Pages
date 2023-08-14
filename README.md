# 半生的猫猫咖啡馆

Wish all my friends live long and prosper.

## todo

- [ ] `code block`优化
  - [x] 使用hugo推荐的`chroma`作为代码块高亮方案
  - [ ] 使得通过配置文件指定代码块样式成为可能，对应`chroma`支持的样式[列表](https://xyproto.github.io/splash/docs/all.html)
  - [x] 从项目中移除`highlighJS`的本地高亮支持，参考项目[papermodX](https://github.com/reorx/hugo-PaperModX)
  
- [x] 优化图片显示和排版
  - [x] 解决图片相对路径插入
  - [ ] 实现插入图片大小限制约束功能
  - [ ] 实现图注功能
- [x] `latex`渲染支持，`latex block`美化
- [ ] `logo`美化字体表现（可exclusive实现）
- [ ] 页面整体风格和配色优化：![#5BCEFA](https://placehold.co/15x15/5BCEFA/5BCEFA.png)、![#FFF](https://placehold.co/15x15/FFF/FFF.png)、![#F5A9B8](https://placehold.co/15x15/F5A9B8/F5A9B8.png)
- [ ] 页面分级（主页=>博客=>文章）功能修复
- [ ] 搜索功能修复
- [ ] 个人简介页重新设计（中文、英文），publication
- [x] 优化正文字体表现，包括中文、英文混排，加粗、斜体、`block`以及`inline tex`混排时的表现
- [ ] 给post增加创建时间、最近修改时间显示功能
- [x] 设计新的浮动清爽`Toc`，随着阅读进度始终保持在主要文章左侧
- [ ] 设计友链页面，添加友链
- [ ] 添加[InstantClick](http://instantclick.io/)机制用于页面的预加载
- [x] 添加[CloudFlare](https://www.cloudflare.com/)CDN功能用于网站分发、入墙和流量安全
- [ ] 使用`Hugo`推荐方案[Chroma](https://gohugo.io/content-management/syntax-highlighting)作为代码高亮方案
- [ ] 实现`Toc`跟踪实施阅读章节功能，[参考](https://www.bram.us/2020/01/10/smooth-scrolling-sticky-scrollspy-navigation/)使用纯粹css方案实现当前阅读章节加粗

## bug

- [x] ~~permalink参数配置无效，无法修改post链接格式~~
- [x] ~~post元信息无法显示~~
- [x] ~~代码高亮主题切换功能存在问题，删除`highlightJS`之后代码块背景显示存在问题~~
- [x] 修复`cloudflare`托管主页之后浏览器报错redirect too many time 错误
- [ ] toc暗色模式下`toc-side-left`比`inner`宽一个像素（且背景色没有随暗色模式切换）
