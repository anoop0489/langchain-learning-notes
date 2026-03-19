
# 00. Environment Setup (The Modern Way with uv) ⚡

## 1. Overview: What is uv?
`uv` is a next-generation Python package and project manager written in Rust.
* **For Newbies:** It replaces complex setups (pip, virtualenv, venv, pyenv) with a single tool.
* **For Pros:** It is 10-100x faster than standard tools and handles dependency resolution instantly.
* **Key Feature:** You don't even need to install Python manually; `uv` manages Python versions for you.

## 2. Prerequisites
* **VS Code** (Recommended Editor)
* **Git**: [Download Here](https://git-scm.com/downloads)
* **PowerShell** (Standard on Windows)

## 3. Installation (Windows)

Open your **PowerShell** as Administrator and run the following command to install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

```

*Note: Close and reopen your terminal after installation to ensure the `uv` command works.*

## 4. Setting Up This Course Project

We will use `uv` to initialize this repository as a proper Python project. This creates a standardized structure that works on any machine.

### A. Clone the Repository

```powershell
git clone [https://github.com/anoop0489/langchain-learning-notes.git](https://github.com/anoop0489/langchain-learning-notes.git)
cd langchain-learning-notes

```

### B. Initialize the Project

Instead of manually creating a virtual environment, we tell `uv` to manage this folder.

```powershell
uv init

```

* This creates a `pyproject.toml` file. This file is the modern standard for defining project requirements (replacing `requirements.txt`).

### C. Add Dependencies

We don't use `pip install` anymore. We use `uv add`. This updates our project file AND installs the packages in one step.

```powershell
uv add langchain langchain-openai python-dotenv

```

*(Optional: If you plan to use local models, runs `uv add langchain-ollama`)*

### D. Security Setup (.env)

**NEVER hard-code API keys.**

1. Create a file named `.env` in the root folder.
2. Paste your key inside:

```text
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx

```

3. **Critical:** Open your `.gitignore` file and ensure `.env` is listed there.

## 5. Running Code (The "uv run" Magic)

In the old days, you had to "activate" a virtual environment (`.venv\Scripts\activate`).
With `uv`, you just use `uv run`. It automatically detects the environment, activates it, and runs your script in isolation.

**Create a test file `test_setup.py`:**

```python
import langchain
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print(f"✅ LangChain Version: {langchain.__version__}")
if api_key:
    print(f"✅ API Key Loaded: {api_key[:5]}...")
else:
    print("❌ API Key NOT found.")

```

**Execute it:**

```powershell
uv run test_setup.py

```

## 6. Technical Deep Dive (Interview Q&A)

### Q: Why use `uv init` instead of `python -m venv`?

> **A:** `uv init` creates a **Declarative** setup via `pyproject.toml`. This means the project's requirements are clearly documented in a file. `python -m venv` is imperative and manual; it doesn't track *what* you installed, leading to "it works on my machine" issues later.

### Q: What is the benefit of `uv run`?

> **A:** `uv run` guarantees **Ephemeral Execution**. It ensures the script runs with *exactly* the dependencies defined in the lockfile, preventing pollution from global packages. It also handles the tedious "activation" step automatically.

---

