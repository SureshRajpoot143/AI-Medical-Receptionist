from __future__ import annotations

import csv
import html
import json
import re
import sys
from cgi import FieldStorage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency fallback
    PdfReader = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"


@dataclass
class DocumentChunk:
    source: str
    title: str
    text: str
    fields: dict[str, str] | None = None


class LocalMedicalReceptionist:
    FILE_HINTS: dict[str, set[str]] = {
        "doctors.csv": {"doctor", "doctors", "specialist", "specialists", "consultant", "cardiology", "neurology", "medicine"},
        "departments.csv": {"department", "departments", "ward", "wards", "location", "services"},
        "appointments.csv": {"appointment", "appointments", "booking", "bookings", "schedule", "slot"},
        "patients.csv": {"patient", "patients", "admitted", "waiting", "under"},
        "wards.csv": {"ward", "wards", "bed", "beds", "capacity", "occupied"},
        "beds.csv": {"bed", "beds", "occupied", "available"},
        "pharmacy.csv": {"pharmacy", "medicine", "medicines", "drug", "drugs", "stock"},
        "diagnostics.csv": {"diagnostic", "diagnostics", "test", "tests", "lab", "imaging"},
        "billing.csv": {"billing", "bill", "invoice", "cost", "price", "charges"},
        "staff.csv": {"staff", "nurse", "receptionist", "pharmacist"},
        "hospital_rules.pdf": {"rule", "rules", "policy", "policies", "visiting", "contact", "emergency"},
        "visiting_hours.pdf": {"visiting", "visiting hours", "hours", "visitor"},
        "emergency_contacts.pdf": {"emergency", "contact", "contacts", "phone", "number"},
    }

    def __init__(self, data_dir: Path, upload_dir: Path | None = None) -> None:
        self.data_dir = data_dir
        self.upload_dir = upload_dir
        self.chunks = self._load_documents()

    def _load_documents(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for directory in [self.data_dir, self.upload_dir]:
            if not directory or not directory.exists():
                continue

            for path in sorted(directory.iterdir()):
                if path.suffix.lower() == ".csv":
                    chunks.extend(self._load_csv(path))
                elif path.suffix.lower() == ".pdf":
                    chunks.extend(self._load_pdf(path))
                elif path.suffix.lower() in {".txt", ".md"}:
                    chunks.append(DocumentChunk(path.name, path.stem, path.read_text(encoding="utf-8", errors="ignore")))
        return chunks

    def _load_csv(self, path: Path) -> list[DocumentChunk]:
        rows: list[DocumentChunk] = []
        try:
            with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader, start=1):
                    text = " ".join(f"{key}: {value}" for key, value in row.items() if value not in {None, ""})
                    if text.strip():
                        fields = {key.strip().lower(): str(value).strip() for key, value in row.items() if value not in {None, ""}}
                        rows.append(DocumentChunk(path.name, f"{path.stem} row {index}", text, fields))
        except Exception:
            pass
        return rows

    def _load_pdf(self, path: Path) -> list[DocumentChunk]:
        if PdfReader is None:
            return []
        try:
            reader = PdfReader(str(path))
            chunks: list[DocumentChunk] = []
            for index, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    chunks.append(DocumentChunk(path.name, f"{path.stem} page {index}", text))
            return chunks
        except Exception:
            return []

    def answer(self, question: str) -> str:
        if not question.strip():
            return "Please ask a question about doctors, departments, appointments, wards, beds, billing, pharmacy, diagnostics, or hospital rules."

        self.chunks = self._load_documents()
        if not self.chunks:
            return "I could not find any data files to read. Add CSV or PDF files under the data folder and ask again."

        scored = sorted(
            ((self._score_chunk(question, chunk), chunk) for chunk in self.chunks),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_chunk = scored[0]
        return self._format_answer(question, best_chunk, scored[:3], best_score <= 0)

    def _score(self, question: str, text: str) -> int:
        question_tokens = self._tokens(question)
        text_tokens = self._tokens(text)
        if not question_tokens or not text_tokens:
            return 0
        overlap = len(question_tokens & text_tokens)
        phrase_bonus = 2 if any(token in text.lower() for token in question_tokens) else 0
        return overlap + phrase_bonus

    def _score_chunk(self, question: str, chunk: DocumentChunk) -> int:
        score = self._score(question, chunk.text)
        question_tokens = self._tokens(question)
        file_name = chunk.source.lower()
        title = chunk.title.lower()
        fields = chunk.fields or {}

        for token in question_tokens:
            if token in file_name or token in title:
                score += 4

        keywords = self.FILE_HINTS.get(file_name)
        if keywords:
            score += sum(3 for token in question_tokens if token in keywords)

        for key, value in fields.items():
            key_tokens = self._tokens(key)
            value_tokens = self._tokens(value)
            if question_tokens & key_tokens:
                score += 3
            if question_tokens & value_tokens:
                score += 4
            if any(token in value.lower() for token in question_tokens):
                score += 8

        if fields:
            score += self._structured_field_boost(question_tokens, fields)

        return score

    def _structured_field_boost(self, question_tokens: set[str], fields: dict[str, str]) -> int:
        score = 0
        normalized_question = " ".join(sorted(question_tokens))

        if "doctor" in question_tokens and fields.get("department"):
            score += 2
        if "department" in question_tokens and fields.get("department"):
            score += 4
        if "patient" in question_tokens and fields.get("patient"):
            score += 4
        if "ward" in question_tokens and fields.get("ward"):
            score += 4
        if "bed" in question_tokens and fields.get("bed"):
            score += 4
        if any(token in question_tokens for token in {"stock", "medicine", "pharmacy"}) and fields.get("medicine"):
            score += 4
        if any(token in question_tokens for token in {"bill", "billing", "cost", "price"}) and fields.get("cost"):
            score += 4

        if "show" in question_tokens or "details" in question_tokens or "list" in question_tokens:
            score += 1

        if "cardiology" in normalized_question and fields.get("department", "").lower() == "cardiology":
            score += 6
        if "neurology" in normalized_question and fields.get("department", "").lower() == "neurology":
            score += 6

        return score

    def _tokens(self, value: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 2}

    def _format_answer(self, question: str, best_chunk: DocumentChunk, top_matches: Iterable[tuple[int, DocumentChunk]], weak_match: bool = False) -> str:
        lines = []

        if weak_match:
            lines.append(f"I could not find an exact local match, so I’m using the closest record from {best_chunk.source} ({best_chunk.title}).")
        else:
            lines.append(f"I found a local match in {best_chunk.source} ({best_chunk.title}).")

        lines.append("")
        lines.append(self._best_effort_answer(question, best_chunk))

        if best_chunk.fields:
            lines.append("")
            lines.append("Details:")
            for key, value in best_chunk.fields.items():
                pretty_key = key.replace("_", " ").title()
                lines.append(f"- {pretty_key}: {value}")

        if "rule" in question.lower() or "visiting" in question.lower() or "contact" in question.lower():
            lines.append("")
            lines.append("If you want, I can also search the specific rules or contact file next.")

        lines.append("")
        lines.append("Other close matches:")
        for score, chunk in top_matches:
            lines.append(f"- {chunk.source} / {chunk.title} (score {score})")
        return "\n".join(lines)

    def _best_effort_answer(self, question: str, chunk: DocumentChunk) -> str:
        fields = chunk.fields or {}
        source = chunk.source.lower()
        question_lower = question.lower()

        if source.endswith(".csv") and any(keyword in question_lower for keyword in {"analyze", "analysis", "summary", "overview", "report"}):
            return self._analyze_csv_source(source)

        if source.endswith(".csv") and fields:
            if "show" in question_lower or "details" in question_lower or "list" in question_lower:
                return self._render_row_summary(fields)

        if source == "doctors.csv":
            name = fields.get("name", "Unknown doctor")
            department = fields.get("department")
            availability = fields.get("availability")
            contact = fields.get("contact")
            parts = [f"Doctor: {name}"]
            if department:
                parts.append(f"Department: {department}")
            if availability:
                parts.append(f"Availability: {availability}")
            if contact:
                parts.append(f"Contact: {contact}")
            if "which doctor" in question_lower or "doctor" in question_lower:
                return "The closest doctor record is: " + "; ".join(parts) + "."
            return "; ".join(parts) + "."

        if source == "departments.csv":
            department = fields.get("department", "Department")
            location = fields.get("location")
            services = fields.get("services")
            parts = [f"Department: {department}"]
            if location:
                parts.append(f"Location: {location}")
            if services:
                parts.append(f"Services: {services}")
            return "; ".join(parts) + "."

        if source == "appointments.csv":
            patient = fields.get("patient", "Patient")
            doctor = fields.get("doctor")
            date = fields.get("date")
            time = fields.get("time")
            status = fields.get("status")
            parts = [f"Appointment for {patient}"]
            if doctor:
                parts.append(f"Doctor: {doctor}")
            if date:
                parts.append(f"Date: {date}")
            if time:
                parts.append(f"Time: {time}")
            if status:
                parts.append(f"Status: {status}")
            return "; ".join(parts) + "."

        if source == "wards.csv":
            ward = fields.get("ward", "Ward")
            capacity = fields.get("capacity")
            occupied = fields.get("occupied")
            notes = fields.get("notes")
            parts = [f"Ward: {ward}"]
            if capacity:
                parts.append(f"Capacity: {capacity}")
            if occupied:
                parts.append(f"Occupied: {occupied}")
            if notes:
                parts.append(f"Notes: {notes}")
            return "; ".join(parts) + "."

        if source == "beds.csv":
            bed = fields.get("bed", "Bed")
            ward = fields.get("ward")
            status = fields.get("status")
            patient = fields.get("patient")
            parts = [f"Bed: {bed}"]
            if ward:
                parts.append(f"Ward: {ward}")
            if status:
                parts.append(f"Status: {status}")
            if patient:
                parts.append(f"Patient: {patient}")
            return "; ".join(parts) + "."

        if source == "pharmacy.csv":
            medicine = fields.get("medicine", "Medicine")
            stock = fields.get("stock")
            use = fields.get("use")
            parts = [f"Medicine: {medicine}"]
            if stock:
                parts.append(f"Stock: {stock}")
            if use:
                parts.append(f"Use: {use}")
            return "; ".join(parts) + "."

        if source == "diagnostics.csv":
            test = fields.get("test", "Test")
            department = fields.get("department")
            turnaround = fields.get("turnaround")
            parts = [f"Test: {test}"]
            if department:
                parts.append(f"Department: {department}")
            if turnaround:
                parts.append(f"Turnaround: {turnaround}")
            return "; ".join(parts) + "."

        if source == "billing.csv":
            service = fields.get("service", "Service")
            cost = fields.get("cost")
            currency = fields.get("currency")
            parts = [f"Service: {service}"]
            if cost and currency:
                parts.append(f"Cost: {cost} {currency}")
            elif cost:
                parts.append(f"Cost: {cost}")
            return "; ".join(parts) + "."

        if fields:
            return "; ".join(f"{key.title()}: {value}" for key, value in fields.items()) + "."

        return self._summarize(chunk.text)

    def _analyze_csv_source(self, source: str) -> str:
        rows = [chunk.fields or {} for chunk in self.chunks if chunk.source.lower() == source and chunk.fields]
        if not rows:
            return f"I found {source}, but it does not contain structured rows I can analyze."

        columns = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)

        row_count = len(rows)
        summary_bits = [f"File: {source}", f"Rows: {row_count}", f"Columns: {', '.join(columns)}"]

        if any("department" in row for row in rows):
            departments = sorted({row.get("department", "").strip() for row in rows if row.get("department")})
            if departments:
                summary_bits.append(f"Departments: {', '.join(departments)}")

        if any("status" in row for row in rows):
            statuses = sorted({row.get("status", "").strip() for row in rows if row.get("status")})
            if statuses:
                summary_bits.append(f"Statuses: {', '.join(statuses)}")

        return "; ".join(summary_bits) + "."

    def _render_row_summary(self, fields: dict[str, str]) -> str:
        preferred_order = ["name", "department", "role", "patient", "doctor", "date", "time", "status", "ward", "bed", "medicine", "test", "service", "availability", "location", "contact", "stock", "cost", "currency", "use", "turnaround", "notes"]
        seen: set[str] = set()
        parts: list[str] = []

        for key in preferred_order:
            if key in fields:
                parts.append(f"{key.replace('_', ' ').title()}: {fields[key]}")
                seen.add(key)

        for key, value in fields.items():
            if key not in seen:
                parts.append(f"{key.replace('_', ' ').title()}: {value}")

        return "; ".join(parts) + "."


def build_page_html() -> str:
        return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI Medical Receptionist</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #f4efe6;
            --bg-2: #e4efe8;
            --card: rgba(255, 255, 255, 0.82);
            --text: #122016;
            --muted: #5a6a61;
            --accent: #0f766e;
            --accent-2: #1f8b63;
            --border: rgba(18, 32, 22, 0.12);
            --shadow: 0 24px 60px rgba(15, 22, 18, 0.14);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            min-height: 100vh;
            background:
                radial-gradient(circle at top left, rgba(31, 139, 99, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.14), transparent 26%),
                linear-gradient(135deg, var(--bg), var(--bg-2));
        }

        .shell {
            width: min(1120px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 28px 0 40px;
        }

        .hero {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 20px;
            align-items: stretch;
            margin-bottom: 18px;
        }

        .panel, .chat-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }

        .panel {
            padding: 28px;
            position: relative;
            overflow: hidden;
        }

        .panel::after {
            content: "";
            position: absolute;
            inset: auto -60px -80px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.18), transparent 68%);
            pointer-events: none;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.1);
            color: var(--accent);
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        h1 {
            margin: 16px 0 10px;
            font-size: clamp(2.2rem, 4vw, 4.2rem);
            line-height: 0.95;
            letter-spacing: -0.04em;
        }

        .lede {
            margin: 0;
            max-width: 60ch;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.6;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 24px;
        }

        .stat {
            padding: 16px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.5);
            border: 1px solid rgba(18, 32, 22, 0.08);
        }

        .stat strong {
            display: block;
            font-size: 1.4rem;
            margin-bottom: 4px;
        }

        .stat span {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .chat-card {
            display: flex;
            flex-direction: column;
            min-height: 68vh;
            overflow: hidden;
        }

        .chat-head {
            padding: 20px 22px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: center;
            background: rgba(255,255,255,0.45);
        }

        .chat-title {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .chat-title strong {
            font-size: 1.05rem;
        }

        .chat-title span,
        .hint {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .status {
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(31, 139, 99, 0.12);
            color: var(--accent-2);
            font-weight: 600;
            font-size: 0.88rem;
            white-space: nowrap;
        }

        .messages {
            flex: 1;
            padding: 22px;
            overflow: auto;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .msg {
            max-width: 82%;
            padding: 14px 16px;
            border-radius: 18px;
            border: 1px solid rgba(18, 32, 22, 0.08);
            line-height: 1.55;
            white-space: pre-wrap;
            animation: rise 180ms ease-out;
        }

        .msg.user {
            margin-left: auto;
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.98), rgba(31, 139, 99, 0.94));
            color: white;
            border-bottom-right-radius: 6px;
        }

        .msg.bot {
            background: rgba(255, 255, 255, 0.78);
            border-bottom-left-radius: 6px;
        }

        .composer {
            padding: 18px 18px 20px;
            border-top: 1px solid var(--border);
            background: rgba(255,255,255,0.55);
        }

        .form {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }

        textarea {
            flex: 1;
            resize: none;
            min-height: 54px;
            max-height: 140px;
            border-radius: 16px;
            border: 1px solid rgba(18, 32, 22, 0.16);
            padding: 14px 16px;
            font: inherit;
            color: var(--text);
            background: rgba(255,255,255,0.9);
            outline: none;
        }

        textarea:focus {
            border-color: rgba(15, 118, 110, 0.5);
            box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
        }

        button {
            border: 0;
            border-radius: 16px;
            padding: 14px 18px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: white;
            font: inherit;
            font-weight: 700;
            cursor: pointer;
            min-width: 112px;
            box-shadow: 0 14px 30px rgba(15, 118, 110, 0.2);
        }

        button:disabled {
            opacity: 0.7;
            cursor: wait;
        }

        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }

        .upload-panel {
            margin-top: 18px;
            padding: 16px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.52);
            border: 1px solid rgba(18, 32, 22, 0.08);
            position: relative;
            z-index: 1;
        }

        .upload-row {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        .upload-row input[type="file"] {
            flex: 1;
            min-width: 240px;
            padding: 10px;
            border-radius: 14px;
            border: 1px dashed rgba(18, 32, 22, 0.2);
            background: rgba(255, 255, 255, 0.7);
        }

        .small-button {
            min-width: 140px;
        }

        .file-list {
            margin: 12px 0 0;
            padding: 0;
            list-style: none;
            display: grid;
            gap: 8px;
        }

        .file-list li {
            padding: 10px 12px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(18, 32, 22, 0.08);
            color: var(--muted);
            font-size: 0.92rem;
        }

        .file-list strong {
            color: var(--text);
        }

        .chip {
            border: 1px solid rgba(18, 32, 22, 0.12);
            background: rgba(255,255,255,0.6);
            color: var(--text);
            border-radius: 999px;
            padding: 10px 14px;
            font-size: 0.92rem;
            cursor: pointer;
        }

        .chip:hover {
            border-color: rgba(15, 118, 110, 0.3);
        }

        @keyframes rise {
            from { transform: translateY(6px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @media (max-width: 900px) {
            .hero { grid-template-columns: 1fr; }
            .chat-card { min-height: 60vh; }
            .stats { grid-template-columns: 1fr; }
            .msg { max-width: 94%; }
            .form { flex-direction: column; }
            button { width: 100%; }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <div class="panel">
                <div class="eyebrow">Local file-powered agent</div>
                <h1>AI Medical Receptionist</h1>
                <p class="lede">Ask about doctors, departments, appointments, wards, beds, billing, diagnostics, or hospital rules. The assistant reads from the CSV and PDF files stored in the <strong>data</strong> folder, so no external API is needed.</p>
                <div class="stats">
                    <div class="stat"><strong>CSV + PDF</strong><span>Answers are pulled from local hospital data files.</span></div>
                    <div class="stat"><strong>Fast</strong><span>No model calls or API keys required.</span></div>
                    <div class="stat"><strong>Structured</strong><span>Row details are shown directly in the response.</span></div>
                </div>
                <div class="suggestions">
                    <button class="chip" data-question="Which doctor is in Cardiology?">Cardiology doctor</button>
                    <button class="chip" data-question="Show me the billing costs">Billing costs</button>
                    <button class="chip" data-question="What are the visiting hours?">Visiting hours</button>
                    <button class="chip" data-question="Show available beds">Available beds</button>
                </div>
                <div class="upload-panel">
                    <strong>Upload your own data</strong>
                    <div class="hint">Add a CSV, PDF, TXT, or MD file. The assistant will read it together with the built-in hospital files.</div>
                    <div class="upload-row">
                        <input id="uploadFiles" type="file" multiple accept=".csv,.pdf,.txt,.md" />
                        <button id="uploadButton" class="small-button">Upload Files</button>
                    </div>
                    <ul id="fileList" class="file-list"></ul>
                </div>
            </div>
            <div class="chat-card">
                <div class="chat-head">
                    <div class="chat-title">
                        <strong>Reception Desk</strong>
                        <span>Ask a question and get a response from the local files.</span>
                    </div>
                    <div class="status" id="status">Ready</div>
                </div>
                <div class="messages" id="messages"></div>
                <div class="composer">
                    <div class="form">
                        <textarea id="question" placeholder="Type your question..." rows="2"></textarea>
                        <button id="send">Ask</button>
                    </div>
                    <div class="hint">Press Enter to send. Shift+Enter for a new line.</div>
                </div>
            </div>
        </section>
    </main>

    <script>
        const messages = document.getElementById('messages');
        const question = document.getElementById('question');
        const send = document.getElementById('send');
        const status = document.getElementById('status');
        const uploadFiles = document.getElementById('uploadFiles');
        const uploadButton = document.getElementById('uploadButton');
        const fileList = document.getElementById('fileList');

        function addMessage(text, who) {
            const bubble = document.createElement('div');
            bubble.className = `msg ${who}`;
            bubble.textContent = text;
            messages.appendChild(bubble);
            messages.scrollTop = messages.scrollHeight;
            return bubble;
        }

        function renderFiles(files) {
            fileList.innerHTML = '';
            if (!files.length) {
                const item = document.createElement('li');
                item.textContent = 'No files loaded yet.';
                fileList.appendChild(item);
                return;
            }

            for (const file of files) {
                const item = document.createElement('li');
                item.innerHTML = `<strong>${file.name}</strong> <span>(${file.type})</span>`;
                fileList.appendChild(item);
            }
        }

        async function refreshFiles() {
            try {
                const response = await fetch('/files');
                const data = await response.json();
                renderFiles(data.files || []);
            } catch (error) {
                renderFiles([]);
            }
        }

        async function ask() {
            const value = question.value.trim();
            if (!value) return;

            addMessage(value, 'user');
            question.value = '';
            question.style.height = 'auto';
            send.disabled = true;
            status.textContent = 'Thinking';

            const typing = addMessage('Looking up the local files...', 'bot');

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: value })
                });
                const data = await response.json();
                typing.textContent = data.answer;
                status.textContent = 'Ready';
            } catch (error) {
                typing.textContent = 'Something went wrong while reading the local files.';
                status.textContent = 'Error';
            } finally {
                send.disabled = false;
                question.focus();
            }
        }

        async function uploadSelectedFiles() {
            const files = uploadFiles.files;
            if (!files || !files.length) {
                status.textContent = 'Choose files first';
                return;
            }

            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }

            uploadButton.disabled = true;
            status.textContent = 'Uploading';

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                });
                const data = await response.json();
                status.textContent = data.uploaded && data.uploaded.length ? 'Files uploaded' : 'No valid files uploaded';
                uploadFiles.value = '';
                await refreshFiles();
            } catch (error) {
                status.textContent = 'Upload failed';
            } finally {
                uploadButton.disabled = false;
            }
        }

        send.addEventListener('click', ask);
        uploadButton.addEventListener('click', uploadSelectedFiles);
        question.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                ask();
            }
        });

        document.querySelectorAll('[data-question]').forEach((button) => {
            button.addEventListener('click', () => {
                question.value = button.dataset.question;
                ask();
            });
        });

        addMessage('Hello. I can read the local hospital files and answer questions from them.', 'bot');
        refreshFiles();
        question.focus();
    </script>
</body>
</html>"""


class ReceptionHTTPRequestHandler(BaseHTTPRequestHandler):
    assistant = LocalMedicalReceptionist(DATA_DIR, UPLOADS_DIR)

    def _reload_assistant(self) -> None:
        self.__class__.assistant = LocalMedicalReceptionist(DATA_DIR, UPLOADS_DIR)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            body = build_page_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/health":
            payload = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed.path == "/files":
            files = []
            for directory, label in [(DATA_DIR, "built-in"), (UPLOADS_DIR, "uploaded")]:
                if directory.exists():
                    for path in sorted(directory.iterdir()):
                        if path.is_file() and path.suffix.lower() in {".csv", ".pdf", ".txt", ".md"}:
                            files.append({"name": path.name, "type": label})
            body = json.dumps({"files": files}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/upload":
            self._handle_upload()
            return

        if parsed.path != "/ask":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="ignore")
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            payload = {}

        answer = self.assistant.answer(str(payload.get("question", "")))
        body = json.dumps({"answer": answer}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_error(400, "Expected multipart form data")
            return

        UPLOADS_DIR.mkdir(exist_ok=True)
        form = FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
        )
        uploaded = form["files"] if "files" in form else []
        if not isinstance(uploaded, list):
            uploaded = [uploaded]

        saved_files = []
        for item in uploaded:
            filename = Path(item.filename or "").name
            if not filename:
                continue
            suffix = Path(filename).suffix.lower()
            if suffix not in {".csv", ".pdf", ".txt", ".md"}:
                continue
            target = UPLOADS_DIR / filename
            with target.open("wb") as handle:
                handle.write(item.file.read())
            saved_files.append(filename)

        self._reload_assistant()
        body = json.dumps({"uploaded": saved_files}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    assistant = LocalMedicalReceptionist(DATA_DIR)

    if len(sys.argv) > 1 and sys.argv[1] in {"--web", "web"}:
        port = 8000
        if len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                pass
        server = ThreadingHTTPServer(("127.0.0.1", port), ReceptionHTTPRequestHandler)
        print(f"Serving the frontend at http://127.0.0.1:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")
        finally:
            server.server_close()
        return 0

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(assistant.answer(question))
        return 0

    print("Local AI Medical Receptionist")
    print("Type a question and I will answer from the files in ./data. Type 'exit' to quit.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError:
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        print(f"\nAssistant:\n{assistant.answer(question)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
