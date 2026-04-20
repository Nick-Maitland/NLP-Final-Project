PYTHON_BOOTSTRAP ?= python3
RUN_LIVE ?= 0
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip
VENV_PYTEST := $(VENV_BIN)/pytest
VALIDATE_OPENAI_ARGS := $(if $(filter 1,$(RUN_LIVE)),--run-live,)
PIP_INSTALL_FLAGS := --disable-pip-version-check --retries 1 --timeout 5
HOST_PYTEST := $(shell command -v pytest 2>/dev/null)

.PHONY: setup-lite setup-full smoke evaluate-offline compare-offline compare-full validate-openai test clean package

$(VENV_PYTHON):
	$(PYTHON_BOOTSTRAP) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m ensurepip --upgrade
	-$(VENV_PIP) install --upgrade pip setuptools wheel

setup-lite: $(VENV_PYTHON)
	@if [ -x "$(VENV_PYTEST)" ]; then \
		echo "[setup-lite] pytest already available in $(VENV_DIR); skipping lite dependency install."; \
	elif [ -n "$(HOST_PYTEST)" ]; then \
		echo "[setup-lite] pytest already available on PATH at $(HOST_PYTEST); skipping lite dependency install."; \
	elif ! $(VENV_PIP) install $(PIP_INSTALL_FLAGS) -r requirements-lite.txt; then \
		echo "[setup-lite] warning: could not install requirements-lite.txt in this environment; continuing with the offline-safe local fallback path."; \
	fi

setup-full: $(VENV_PYTHON)
	$(VENV_PIP) install $(PIP_INSTALL_FLAGS) -r requirements.txt
	$(VENV_PIP) show wheel >/dev/null 2>&1 || $(VENV_PIP) install wheel
	$(VENV_PIP) show ragfaq >/dev/null 2>&1 || $(VENV_PIP) install --no-build-isolation -e .

smoke: setup-lite
	PYTEST_BIN="$$(if [ -x "$(VENV_PYTEST)" ]; then printf '%s' "$(VENV_PYTEST)"; elif command -v pytest >/dev/null 2>&1; then command -v pytest; fi)" && \
	if [ -z "$$PYTEST_BIN" ]; then \
		echo "[smoke] error: pytest is unavailable. Install requirements-lite.txt or run from an environment that already provides pytest." >&2; \
		exit 1; \
	fi && \
	OPENAI_API_KEY= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHON="$(VENV_PYTHON)" PYTEST="$$PYTEST_BIN" bash scripts/local_smoke_test.sh

evaluate-offline: setup-lite
	OPENAI_API_KEY= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(VENV_PYTHON) rag_system.py --build-index --backend tfidf
	OPENAI_API_KEY= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(VENV_PYTHON) rag_system.py --evaluate --backend tfidf --offline

compare-offline: setup-lite
	OPENAI_API_KEY= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(VENV_PYTHON) scripts/run_backend_comparison.py --offline-only

compare-full: setup-lite
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(VENV_PYTHON) scripts/run_backend_comparison.py --include-openai

validate-openai: setup-lite
	$(VENV_PYTHON) scripts/validate_openai_path.py $(VALIDATE_OPENAI_ARGS)

test: setup-lite
	PYTEST_BIN="$$(if [ -x "$(VENV_PYTEST)" ]; then printf '%s' "$(VENV_PYTEST)"; elif command -v pytest >/dev/null 2>&1; then command -v pytest; fi)" && \
	if [ -z "$$PYTEST_BIN" ]; then \
		echo "[test] error: pytest is unavailable. Install requirements-lite.txt or run from an environment that already provides pytest." >&2; \
		exit 1; \
	fi && \
	OPENAI_API_KEY= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$$PYTEST_BIN" -q

clean:
	rm -rf $(VENV_DIR) .pytest_cache build dist .ragfaq *.egg-info src/*.egg-info

package: setup-lite
	if $(VENV_PYTHON) -m pip show build >/dev/null 2>&1; then \
		$(VENV_PYTHON) -m build --no-isolation; \
	else \
		$(VENV_PYTHON) setup.py sdist bdist_wheel; \
	fi
	$(VENV_PYTHON) scripts/package_submission.py
