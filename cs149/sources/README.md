# Fall 2023 原始资料索引

使用 [manifest.json](./manifest.json) 中的每讲 Slides 与视频地址。2023 年有 **18 个主题课（对应 18 份 Slides）与 1 次 Midterm Review（录播第 14 段）**；因此播放列表的第 15–19 段对应官方课程 L14–L18。

## 数据可信度

- **Slides：** 18 份 PDF 均通过 Stanford Fall 2023 官方页面逐讲核对过下载地址。
- **视频：** 18 个主题讲次的 YouTube ID 已与 2023 课程资源索引对应；期中复习另列。
- **Transcript：** 尚未取得可完整逐段核对的字幕文本；网络上部分自称 Transcript 的页面只提供几百字符的预览。完整字幕必须通过 YouTube 的 transcript 功能或本地字幕文件提供后才能进行逐段核对。
- **教材章节：** 仓库只把真正写好的文本标为 `draft_from_slides_with_video_reference`；该状态**不代表已经逐字读取或校对整场视频**。

## 本地字幕建议格式

```text
cs149/sources/raw/
  01.en.vtt
  02.en.vtt
  ...
  18.en.vtt
  review.en.vtt
```

`raw/` 与 `packets/` 已加入忽略规则，避免把完整第三方字幕、PDF 内容直接公开到 GitHub。字幕文件仅供合法个人学习和事实核对，教材正文采用原创表述并标注原始来源。

## 具体怎么做？

阅读 [tools/README.md](../tools/README.md)：从公开视频尝试下载字幕，与 Slides PDF 组合成带页码、带时间戳的本地检索材料。文字稿如果存在缺漏或自动字幕术语误识别，应回放视频核对，而不是假定它准确。

[Stanford Fall 2023 官方课程表](https://gfxcourses.stanford.edu/cs149/fall23/)。
