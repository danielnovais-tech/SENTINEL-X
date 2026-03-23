# Contributing to SENTINEL-X

Thank you for your interest in contributing to SENTINEL-X!  This document
explains the development workflow, code style expectations, and test
requirements.

SENTINEL-X is released under the [Apache License 2.0](LICENSE).  By
submitting a pull request you agree that your contribution may be distributed
under those terms.

---

## Table of contents

- [Getting started](#getting-started)
- [Reporting bugs](#reporting-bugs)
- [Suggesting enhancements](#suggesting-enhancements)
- [Development workflow](#development-workflow)
- [Code style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull-request checklist](#pull-request-checklist)

---

## Getting started

```bash
# Clone the repository
git clone https://github.com/danielnovais-tech/SENTINEL-X.git
cd SENTINEL-X

# Install in editable mode with dev extras
pip install -e ".[dev,yaml]"

# Run the full test suite to confirm a clean baseline
python -m pytest tests/ -v
```

Python 3.10+ is required.  TensorFlow ≥ 2.12 is the only non-trivial
dependency; a CUDA-capable GPU is optional (CPU training works fine for the
default scenario sizes).

---

## Reporting bugs

Please open a [GitHub Issue](https://github.com/danielnovais-tech/SENTINEL-X/issues)
and include:

1. A minimal reproducible example.
2. The full traceback.
3. Your Python and TensorFlow versions (`python -m sentinel_x.config --version`).
4. OS and platform (especially relevant for hardware driver issues).

---

## Suggesting enhancements

Open a GitHub Issue with the label **enhancement** and describe:

1. The problem you are trying to solve.
2. The proposed solution and its trade-offs.
3. Any relevant prior art or references.

---

## Development workflow

1. Fork the repository and create a feature branch:

   ```bash
   git checkout -b feature/my-awesome-feature
   ```

2. Make your changes in small, focused commits.

3. Add or update tests (see [Testing](#testing) below).

4. Add or update documentation in `docs/` if your change affects the public
   API or user-facing behaviour.

5. Push and open a pull request against `main`.

---

## Code style

- **Python**: follow [PEP 8](https://peps.python.org/pep-0008/) and
  [PEP 257](https://peps.python.org/pep-0257/).
- Line length: 99 characters max.
- Use type annotations for all new public functions.
- No unused imports; no bare `except:` clauses.
- Prefer `numpy` vectorised operations over Python loops in hot paths.
- Comment non-obvious algorithmic choices; avoid redundant comments.

---

## Testing

All code changes must be accompanied by tests consistent with the existing
`tests/test_sentinel_x.py` style.  Requirements:

- One test **class** per new public class or module (e.g., `class TestFoo:`).
- Tests must be **self-contained** and **deterministic** (seed RNGs where
  necessary).
- Avoid slow tests (> 10 s each); use small model sizes and few episodes.
- Hardware-dependent paths must have a simulation fallback and skip cleanly
  when hardware is absent (`pytest.mark.skipif` or `try/except ImportError`).

Run tests:

```bash
# Full suite
python -m pytest tests/ -v

# Specific class
python -m pytest tests/ -k TestFormationController -v

# With coverage (optional)
pip install pytest-cov
python -m pytest tests/ --cov=sentinel_x --cov-report=term-missing
```

The CI pipeline (`.github/workflows/ci.yml`) must pass before a PR can be
merged.

---

## Documentation

Public API additions must include:

- Module-level docstring with purpose, parameters, and examples.
- Class/function docstrings (NumPy docstring style).
- A `docs/` Markdown file if the addition warrants a dedicated guide
  (e.g., a new integration adapter or coordination protocol).
- Update the **Package Structure** and **Documentation** sections of
  `README.md`.

---

## Pull-request checklist

Before requesting review, ensure:

- [ ] All existing tests pass (`python -m pytest tests/ -v`).
- [ ] New functionality is covered by new tests.
- [ ] Public API is documented (docstrings + `docs/` guide if needed).
- [ ] `README.md` Package Structure / Documentation sections are updated.
- [ ] No secrets, credentials, or large binary files are committed.
- [ ] `pyproject.toml` optional extras are updated if new optional
      dependencies were added.

---

## Contact

Questions?  Open a GitHub Issue or start a Discussion.  For security
vulnerabilities, please report them privately via GitHub's
[security advisory](https://github.com/danielnovais-tech/SENTINEL-X/security/advisories/new)
feature.

---

*Thank you for contributing to SENTINEL-X!*
