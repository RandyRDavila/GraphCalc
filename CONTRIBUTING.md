# Contributing to GraphCalc

First off, thank you for taking the time to contribute to **GraphCalc**! This project is built with care to advance open, reproducible research in graph theory and mathematical discovery. Whether you’re fixing a typo, improving documentation, suggesting a feature, or contributing code—you’re helping build something valuable.

## 📬 Before You Begin

Please read the [README](./README.md) for an overview of the project. For questions or private suggestions, you can also reach out via email: **<rrd6@rice.edu>**.

---

## 💡 Ways to Contribute

### 🐞 Report Issues or Bugs

If you encounter unexpected behavior or incorrect results, please open an issue:

- Use the **Bug Report** template.
- Include a minimal reproducible example and environment info.

### 🚀 Suggest Features

We welcome proposals for new functionality, heuristics, or performance improvements. Use the **Feature Request** template and provide examples or motivating use cases.

### 🧪 Improve Documentation

Typos, unclear explanations, or better examples? Submit a PR with your improvements. You can run the docs locally with:

```bash
make html
open build/html/index.html
```

### 🧩 Contribute Code

Pull requests are welcome! Before submitting code:

- Add unit tests for new features.

- Run pytests

- Include a NumPy-style docstring and usage examples.

- Clone and install in editable mode:

```bash
git clone https://github.com/your-username/graphcalc.git
cd graphcalc
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

### 📁 Project Structure

```bash
src/                  # Core logic and invariants
tests/                # Unit tests
docs/                 # Sphinx documentation
examples/             # TODO: Jupyter notebooks and demos
```

### 5. 🙏 Thanks

Every contribution—big or small—helps improve GraphCalc. Your curiosity, creativity, and insight matter.
