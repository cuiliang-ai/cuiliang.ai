---
title: {{ replace .File.ContentBaseName "-" " " | title | jsonify }}
date: {{ .Date }}
draft: true
summary: "" # 首页和列表页摘要，说明文章的问题与读者收获
description: "" # 搜索引擎和分享预览描述，可与 summary 相同
tags: [] # 发布前从 hugo.yaml 的 params.canonicalTags 中选择 1–4 个标签
categories: [] # 技术文章通常使用 ["AI Agent Engineering"]
series: [] # 属于专题时填写已有专题名称
ShowToc: true
TocOpen: false
---
