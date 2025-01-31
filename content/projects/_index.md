---
title: "Projects"
date: 2025-01-31
type: "section"
layout: "list"
summary: "Explore my AI and Machine Learning projects."
_build:
  list: always
  render: always
cascade:
  _target:
    path: "projects/**"
  type: "projects"
  layout: "single"
  _build:
    list: always
    render: always
---