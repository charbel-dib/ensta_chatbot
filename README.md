# ACA-ENSTA - ENSTA Campus Assistant 
## Quick Start

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --log-level info --no-access-log
```
This repository contains a complete **end-to-end prototype** of an ENSTA campus assistant based on **Retrieval-Augmented Generation (RAG)**:
- an **offline pipeline** that builds a knowledge base (**web + PDFs + XLSX → FAISS index**),
- an **online FastAPI service** that answers with **grounded citations** and a **safe refusal + escalation** workflow,
- a **web UI + embeddable widget** for demonstration,
- **continuous updates** with a daily 03:00 web rebuild and **hot reload** (no server restart).

---

## 1) What’s inside 

### Root
- `app.py` — FastAPI service (RAG + sessions + refusal + email escalation + hot reload)
- `.env.example` — environment template
- `requirements.txt` — Python dependencies
- `cloudflared.exe`, `cloudflared.log`, `config.yml`, `cert.pem` — Cloudflare exposure artifacts 
- `CONFIGURATION.md` — extra notes (if used)


### UI
- `templates/chat.html` — main chat interface
- `templates/demo_site.html` — demo webpage
- `static/widget.js` — embeddable widget

### Dataset + pipeline
- `ENSTA_DATASET/01_raw/` — raw extracted data (JSONL): `web_docling.jsonl`, `pdfs_docling.jsonl`, `xlsx_courses.jsonl`, etc.
- `ENSTA_DATASET/02_clean/` — cleaned corpora + reports (`corpus_clean.jsonl`, `clean_report_docling.json`, `*_clean.jsonl`)
- `ENSTA_DATASET/03_chunks/chunks.jsonl` — retrieval units
- `ENSTA_DATASET/04_index/` — FAISS + meta + `reload.flag` + `backups/`
- `ENSTA_DATASET/scripts/` — pipeline scripts + Task Scheduler scripts
- `ENSTA_DATASET/logs/` — daily pipeline logs

### Main scripts (used)
- `ENSTA_DATASET/scripts/crawl_web_docling.py` — Docling web crawl → `01_raw/web_docling.jsonl`
- `ENSTA_DATASET/scripts/doclingp.py` — clean/normalize raw → `02_clean/*_clean.jsonl`
- `ENSTA_DATASET/scripts/merge_corpus.py` — merge → `02_clean/corpus_clean.jsonl`
- `ENSTA_DATASET/scripts/chunk_corpus.py` — chunk → `03_chunks/chunks.jsonl`
- `ENSTA_DATASET/scripts/embed_faiss.py` — embed + FAISS → `04_index/ensta.faiss` + `04_index/ensta_meta.jsonl`
- `ENSTA_DATASET/scripts/run_web_daily_update.py` — daily web rebuild orchestrator
- `ENSTA_DATASET/scripts/run_docs_manual_update.py` — manual docs rebuild orchestrator (PDF/XLSX)
- `ENSTA_DATASET/scripts/ingest_pdfs.py` — ingest PDFs → `01_raw/pdfs_docling.jsonl`
- `ENSTA_DATASET/scripts/ingest_xlsx_courses.py` — ingest XLSX → `01_raw/xlsx_courses.jsonl`
- `ENSTA_DATASET/scripts/install_daily_task.ps1` — Windows Task Scheduler installer (03:00)

(Other experiments are stored under `ENSTA_DATASET/scripts/unused/`.)

---

## 2) Requirements

### OS
- Windows 10/11 recommended (Task Scheduler workflow is included)
- Linux is possible (replace Task Scheduler by cron/systemd)

### Python
- Python **3.10+** (3.11 tested)

### Install dependencies
Create venv, install requirements:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional environment variables to reduce TensorFlow logs (recommended on Windows):
```powershell
$env:TRANSFORMERS_NO_TF="1"
$env:TF_CPP_MIN_LOG_LEVEL="2"
```

---

## 3) Configuration (all settings)

### 3.1 Create `.env`
Copy the template:

```powershell
copy .env.example .env
```

### 3.2 Required environment variables
These are required for the server to answer questions:

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `ENSTA_DATASET` | Path to `ENSTA_DATASET` directory |
| `FAISS_INDEX_PATH` | Path to `ENSTA_DATASET/04_index/ensta.faiss` |
| `META_JSONL_PATH` | Path to `ENSTA_DATASET/04_index/ensta_meta.jsonl` |

Example (Windows):
```env
OPENAI_API_KEY=sk-...

ENSTA_DATASET=C:\Users\charb\OneDrive - ENSTA\Documents\Projet 3A\to github\ENSTA_DATASET
FAISS_INDEX_PATH=C:\Users\charb\OneDrive - ENSTA\Documents\Projet 3A\to github\ENSTA_DATASET\04_index\ensta.faiss
META_JSONL_PATH=C:\Users\charb\OneDrive - ENSTA\Documents\Projet 3A\to github\ENSTA_DATASET\04_index\ensta_meta.jsonl
```

### 3.3 RAG settings
| Variable | Default | Meaning |
|---|---:|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | model for answer generation |
| `EMB_MODEL` | `intfloat/multilingual-e5-base` | query embedding model |
| `QUERY_PREFIX` | `query:` | E5 query prefix |
| `TOP_K` | `8` | retrieved chunks |
| `MIN_SCORE` | `0.08` | similarity threshold (refuse below) |
| `OPENAI_CONCURRENCY` | `6` | max parallel LLM calls |

### 3.4 Hot reload (index update without restart)
| Variable | Default | Meaning |
|---|---:|---|
| `RELOAD_FLAG_PATH` | `ENSTA_DATASET/04_index/reload.flag` | file touched after rebuild |
| `RELOAD_EVERY_SECONDS` | `30` | polling interval |

Behavior:
- Offline pipeline rebuilds `ensta.faiss` + `ensta_meta.jsonl`
- Pipeline writes/updates `reload.flag`
- `app.py` detects mtime change → reloads index + meta in memory (no restart)

### 3.5 Sessions / throttling
| Variable | Default | Meaning |
|---|---:|---|
| `TTL_SECONDS` | `120` | idle timeout before close |
| `HISTORY_MAX` | `400` | max events stored per session |
| `RATE_LIMIT_MIN_SECONDS` | `0.25` | anti-spam throttle |

### 3.6 Logging
| Variable | Default | Meaning |
|---|---:|---|
| `APP_LOG_LEVEL` | `INFO` | app log level |
| `SILENCE_UVICORN_LOGS` | `0` | if `1`, reduce uvicorn logs |

Logs:
- `logs/app.log` (rotating)

### 3.7 Email escalation (SMTP)
If you want escalation enabled, configure SMTP:

| Variable | Meaning |
|---|---|
| `SMTP_HOST` | SMTP host (ex: `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port (587 typical) |
| `SMTP_USER` | SMTP username |
| `SMTP_PASS` | SMTP password / app password |
| `SMTP_FROM` | from address |
| `SUPPORT_TO` | destination support mailbox |
| `SUPPORT_SUBJECT_PREFIX` | subject prefix |

Important:
- Never commit secrets
- Gmail requires an **App Password** (2FA)

Example:
```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=yourbot@gmail.com
SMTP_PASS=YOUR_APP_PASSWORD
SMTP_FROM=yourbot@gmail.com

SUPPORT_TO=service@ensta.fr
SUPPORT_SUBJECT_PREFIX=[ENSTA Chatbot - Ticket]
```

---

## 4) Run the FastAPI service

From repository root:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --log-level info --no-access-log
```

Health check:
```bash
curl http://127.0.0.1:8001/health
```

Open:
- Chat UI: `http://127.0.0.1:8001/chat`
- Demo page: `http://127.0.0.1:8001/demo-site`
- Widget JS: `http://127.0.0.1:8001/widget.js`

### Startup note
The first load of `SentenceTransformer` may take ~20–40s on Windows. The server is reachable early, while RAG loads in the background. Check `/health` for `rag_ready=true`.

---

## 5) Offline pipeline (build/rebuild the index)

The pipeline builds:
**(web/pdf/xlsx) → clean → merge → chunk → embed → FAISS + meta**

Scripts are under: `ENSTA_DATASET/scripts/`.

### 5.1 Full build (web + docs + index)
Run in this order:

```bash
python ENSTA_DATASET/scripts/crawl_web_docling.py
python ENSTA_DATASET/scripts/ingest_pdfs.py
python ENSTA_DATASET/scripts/ingest_xlsx_courses.py

python ENSTA_DATASET/scripts/doclingp.py
python ENSTA_DATASET/scripts/merge_corpus.py
python ENSTA_DATASET/scripts/chunk_corpus.py
python ENSTA_DATASET/scripts/embed_faiss.py
```

Outputs:
- `ENSTA_DATASET/01_raw/web_docling.jsonl`
- `ENSTA_DATASET/01_raw/pdfs_docling.jsonl`
- `ENSTA_DATASET/01_raw/xlsx_courses.jsonl`
- `ENSTA_DATASET/02_clean/*_clean.jsonl` + reports
- `ENSTA_DATASET/02_clean/corpus_clean.jsonl`
- `ENSTA_DATASET/03_chunks/chunks.jsonl`
- `ENSTA_DATASET/04_index/ensta.faiss`
- `ENSTA_DATASET/04_index/ensta_meta.jsonl`

### 5.2 Daily web rebuild (recommended method)
Run:
```bash
python ENSTA_DATASET/scripts/run_web_daily_update.py
```

What it does:
- web crawl (Docling)
- clean
- merge with existing PDF/XLSX clean outputs
- chunk
- backup old index (`ENSTA_DATASET/04_index/backups/<timestamp>/`)
- embed + new FAISS
- atomic swap of index/meta
- write/update `reload.flag`

Logs:
- `ENSTA_DATASET/logs/web_daily_YYYYMMDD_HHMMSS.log`

### 5.3 Manual docs rebuild (PDF/XLSX)
When you add/modify PDFs or XLSX:
```bash
python ENSTA_DATASET/scripts/run_docs_manual_update.py
```

What it does:
- ingest PDFs + XLSX
- clean
- merge
- chunk
- backup
- embed + rebuild FAISS
- write/update `reload.flag`

---

## 6) Continuous updates (Windows Task Scheduler)

A daily task runs the web pipeline every day at **03:00**.

### 6.1 Install task
```powershell
powershell -ExecutionPolicy Bypass -File .\ENSTA_DATASET\scripts\install_daily_task.ps1
```

Task name:
- `ACAENSTA_WebDailyUpdate`

### 6.2 Status / enable / run now
```powershell
Get-ScheduledTask -TaskName "ACAENSTA_WebDailyUpdate" | Select TaskName, State
Enable-ScheduledTask -TaskName "ACAENSTA_WebDailyUpdate"
Start-ScheduledTask -TaskName "ACAENSTA_WebDailyUpdate"
```

### 6.3 Remove and recreate (if you changed paths)
```powershell
Unregister-ScheduledTask -TaskName "ACAENSTA_WebDailyUpdate" -Confirm:$false
powershell -ExecutionPolicy Bypass -File .\ENSTA_DATASET\scripts\install_daily_task.ps1
```

### Troubleshooting
- **“The task is disabled”** → `Enable-ScheduledTask`
- **“The system cannot find the file specified”** → edit `install_daily_task.ps1`:
  - `$ProjectRoot`
  - `$PythonExe` (must point to your venv python)
  - `$ScriptPath`
  Then reinstall.

---

## 7) Runtime behavior (how the assistant works)

### 7.1 RAG flow
For each user message:
1. Handle greeting / meta / gibberish without retrieval
2. Embed query (`EMB_MODEL`, `QUERY_PREFIX`)
3. FAISS search (`TOP_K`)
4. If best score < `MIN_SCORE` → refuse
5. Else → call OpenAI with strict grounding rules and inline citations

### 7.2 Safe refusal + escalation
If refused:
- assistant asks the user for an email address
- if SMTP configured → sends ticket to `SUPPORT_TO` with:
  - session id
  - unanswered question
  - recent transcript excerpt

### 7.3 Hot reload
After daily rebuild:
- pipeline updates `reload.flag`
- server reloads `ensta.faiss` + `ensta_meta.jsonl` in memory
- no restart required

---

## 8) UI + Widget

### 8.1 Chat UI
- `/chat` serves `templates/chat.html`

### 8.2 Demo page
- `/demo-site` serves `templates/demo_site.html`

### 8.3 Widget
- `/widget.js` serves `static/widget.js`

Widget purpose:
- simulate integration on an institutional website without modifying the site layout

---

## 9) Cloudflare Tunnel exposure

You have `cloudflared.exe` at repo root.

### 9.1 Quick tunnel
```powershell
.\cloudflared.exe tunnel --url http://127.0.0.1:8001
```

### 9.2 Known limitation
Some campus networks block Cloudflare tunnel traffic. If blocked:
- test from a hotspot
- or run on a VPS/cloud host

---
