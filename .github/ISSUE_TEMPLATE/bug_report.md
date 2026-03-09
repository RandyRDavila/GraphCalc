---
name: 🐞 Bug Report
about: Report unexpected behavior, incorrect output, or a crash in `graphcalc`
title: "[Bug] "
labels: bug
assignees: randyrdavila
---

## 🧩 What happened?

Describe the issue clearly. What function or feature of `graphcalc` behaved incorrectly?

## 📋 Minimal Example

Please include a minimal, self-contained code snippet that reproduces the issue.

```python
import graphcalc.graphs as gc
import networkx as nx

# Example graph
G = nx.path_graph(4)

# Function call with unexpected result
print(gc.independence_number(G))  # Expected: ..., Got: ...
```

If the issue only occurs for specific graphs or inputs, please attach or describe them clearly.

## ✅ Expected Behavior

What result did you expect from this code or function?

## Actual Behavior

What actually happened instead? If there was an error, paste the full traceback below.

## 🧪 Environment Info

Please complete the following so we can reproduce your environment:

- OS: (e.g., macOS 13.4, Ubuntu 22.04, Windows 11)
- Python version: (e.g., 3.10.6) – run `python --version`
- graphcalc version: (e.g., 0.3.1) – run `pip show graphcalc`
- Backend (if applicable): e.g., PuLP solver, NetworkX, `SimpleGraph`
