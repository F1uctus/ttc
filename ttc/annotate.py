"""Browser-based annotation tool for expanding the test corpus.

Runs the pipeline over a raw text, then serves a single local web page
where each extracted replica carries its predicted speaker; the human
corrects mistakes by clicking a name in the character palette (or keys
1-9), accepts correct predictions with Enter, and saves the result
directly in the corpus format understood by :mod:`ttc.corpus`.

Stdlib only — no dependencies beyond ttc itself.
"""

import json
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ttc.corpus import (
    DELIMITER,
    UNATTRIBUTED,
    normalize_name,
    parse_corpus_content,
    serialize_corpus_file,
)

PALETTE_LIMIT = 15


def build_payload(cc, text: str, prefill: dict[str, str] | None = None) -> dict:
    """Extract replicas + predicted speakers and a character palette.

    ``prefill`` maps replica text -> actor for re-annotation of an
    existing corpus file; it overrides pipeline predictions.
    """
    dialogue = cc.extract_dialogue(text)
    play = cc.connect_play(dialogue)
    doc = dialogue.doc
    # \n -> " " replacement is length-preserving, so span offsets
    # index directly into the original text.
    assert len(doc.text) == len(text), "doc/text offset invariant broken"

    replicas = []
    frequency: Counter = Counter()
    for r, actor in play.lines:
        pred = normalize_name(str(actor)) if actor and len(actor) else UNATTRIBUTED
        if prefill and str(r) in prefill:
            pred = normalize_name(prefill[str(r)])
        replicas.append(
            {
                "start": r.start_char,
                "end": r.end_char,
                "text": str(r),
                "actor": pred,
            }
        )
        if pred != UNATTRIBUTED:
            frequency[pred] += 1

    in_replica = {i for r in play.replicas for i in range(r.start, r.end)}
    for token in doc:
        # candidate names come from author speech: inside a replica a
        # name is usually the addressee, not the speaker
        if token.i in in_replica:
            continue
        if token.pos_ == "PROPN" or token.ent_type_ == "PER":
            name = normalize_name(token.lemma_)
            if name and name not in frequency:
                frequency[name] += 0  # candidate with zero predicted uses

    palette = [name for name, _ in frequency.most_common(PALETTE_LIMIT)]
    return {"text": text, "replicas": replicas, "palette": palette}


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>ttc annotate — {title}</title>
<style>
  :root {{
    color-scheme: light dark;
    --accent: #7c4dff; --ok: #2e7d32; --warn: #b26a00; --bad: #b3261e;
  }}
  body {{ margin: 0; font: 15px/1.55 system-ui, sans-serif; display: flex; }}
  #text {{ flex: 1; padding: 1.2rem 1.5rem; white-space: pre-wrap; overflow-y: auto;
          height: 100vh; box-sizing: border-box; }}
  #side {{ width: 19rem; border-left: 1px solid #8884; padding: 1rem; height: 100vh;
          overflow-y: auto; box-sizing: border-box; position: sticky; top: 0; }}
  .r {{ border-radius: 4px; cursor: pointer; padding: 0 2px;
       background: color-mix(in srgb, var(--warn) 14%, transparent); }}
  .r.confirmed {{ background: color-mix(in srgb, var(--ok) 16%, transparent); }}
  .r.bogus {{ text-decoration: line-through; opacity: .45; }}
  .r.sel {{ outline: 2px solid var(--accent); }}
  .chip {{ font-size: .75em; font-weight: 600; border-radius: 8px; padding: 0 .4em;
          margin-right: .25em; background: var(--accent); color: white;
          user-select: none; white-space: nowrap; }}
  .r.confirmed .chip {{ background: var(--ok); }}
  #palette button {{ display: block; width: 100%; margin: .15rem 0; padding: .3rem .5rem;
                    text-align: left; cursor: pointer; border-radius: 6px;
                    border: 1px solid #8886; background: transparent; font: inherit; }}
  #palette button:hover {{ border-color: var(--accent); }}
  #palette .key {{ opacity: .55; font-size: .8em; margin-right: .4em; }}
  #aliases {{ width: 100%; min-height: 5rem; box-sizing: border-box; font: .85em/1.4 monospace; }}
  #save {{ width: 100%; padding: .5rem; margin-top: .6rem; font: inherit; font-weight: 700;
          cursor: pointer; border-radius: 6px; border: 0; background: var(--accent); color: white; }}
  #progress {{ font-weight: 600; margin: .4rem 0; }}
  #status {{ min-height: 1.4em; font-size: .85em; }}
  .dirty {{ color: var(--warn); }} .saved {{ color: var(--ok); }} .error {{ color: var(--bad); }}
  kbd {{ font-size: .8em; border: 1px solid #8886; border-radius: 3px; padding: 0 .25em; }}
  #newName {{ width: 100%; box-sizing: border-box; padding: .25rem; font: inherit; }}
</style>
</head>
<body>
<div id="text"></div>
<div id="side">
  <div id="progress"></div>
  <div id="palette"></div>
  <input id="newName" placeholder="+ новый персонаж (Enter)">
  <p style="font-size:.8em; opacity:.75">
    <kbd>Enter</kbd> принять и дальше · <kbd>1</kbd>–<kbd>9</kbd> назначить ·
    <kbd>0</kbd> без говорящего · <kbd>x</kbd> не реплика ·
    <kbd>j</kbd>/<kbd>k</kbd> или <kbd>↓</kbd>/<kbd>↑</kbd> навигация
  </p>
  <details open><summary>Алиасы (канон = алиас | алиас)</summary>
    <textarea id="aliases" spellcheck="false"></textarea>
  </details>
  <button id="save">Сохранить (⇧: и выйти)</button>
  <div id="status"></div>
</div>
<script>
const DATA = {payload};
const UNATTRIBUTED = {unattributed};
let sel = 0, dirty = false;
const st = DATA.replicas.map(r => ({{...r, confirmed: false, bogus: false}}));

function esc(s) {{ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;"); }}

function render() {{
  const parts = []; let pos = 0;
  st.forEach((r, i) => {{
    parts.push(esc(DATA.text.slice(pos, r.start)));
    const cls = "r" + (r.confirmed ? " confirmed" : "") + (r.bogus ? " bogus" : "")
              + (i === sel ? " sel" : "");
    parts.push(`<span class="${{cls}}" data-i="${{i}}"><span class="chip">${{esc(r.actor)}}</span>${{esc(DATA.text.slice(r.start, r.end))}}</span>`);
    pos = r.end;
  }});
  parts.push(esc(DATA.text.slice(pos)));
  document.getElementById("text").innerHTML = parts.join("");
  document.querySelectorAll(".r").forEach(el => el.onclick = e => {{
    sel = +el.dataset.i;
    if (e.target.classList.contains("chip")) cycle();
    else render();
  }});
  const done = st.filter(r => r.confirmed || r.bogus).length;
  document.getElementById("progress").textContent = `${{done}}/${{st.length}} подтверждено`;
  document.getElementById("status").innerHTML =
    dirty ? '<span class="dirty">есть несохранённые правки</span>' : "";
  const selEl = document.querySelector(".r.sel");
  if (selEl) selEl.scrollIntoView({{block: "nearest"}});
  renderPalette();
}}

function renderPalette() {{
  const pal = document.getElementById("palette");
  pal.innerHTML = "";
  [...DATA.palette, UNATTRIBUTED].forEach((name, i) => {{
    const b = document.createElement("button");
    const key = i < 9 ? i + 1 : (name === UNATTRIBUTED ? 0 : null);
    b.innerHTML = (key !== null ? `<span class="key">${{key}}</span>` : "") + esc(name);
    b.onclick = () => assign(name);
    pal.appendChild(b);
  }});
}}

function assign(name) {{
  const r = st[sel];
  r.actor = name; r.confirmed = true; r.bogus = false;
  dirty = true; advance();
}}

function cycle() {{
  const r = st[sel];
  const all = [...DATA.palette, UNATTRIBUTED];
  r.actor = all[(all.indexOf(r.actor) + 1) % all.length];
  r.confirmed = true; dirty = true; render();
}}

function advance() {{ if (sel < st.length - 1) sel++; render(); }}

document.getElementById("newName").addEventListener("keydown", e => {{
  if (e.key === "Enter") {{
    const name = e.target.value.trim().toLowerCase();
    if (name) {{ DATA.palette.push(name); assign(name); e.target.value = ""; }}
    e.stopPropagation();
  }}
  e.stopPropagation();
}});
document.getElementById("aliases").addEventListener("keydown", e => e.stopPropagation());
document.getElementById("aliases").addEventListener("input", () => {{ dirty = true; }});

document.addEventListener("keydown", e => {{
  if (e.key === "Enter") {{ st[sel].confirmed = true; dirty = true; advance(); }}
  else if (e.key === "j" || e.key === "ArrowDown") {{ if (sel < st.length - 1) sel++; render(); }}
  else if (e.key === "k" || e.key === "ArrowUp") {{ if (sel > 0) sel--; render(); }}
  else if (e.key === "x") {{ st[sel].bogus = !st[sel].bogus; st[sel].confirmed = false; dirty = true; advance(); }}
  else if (e.key === "0") assign(UNATTRIBUTED);
  else if (/^[1-9]$/.test(e.key)) {{
    const name = [...DATA.palette, UNATTRIBUTED][+e.key - 1];
    if (name) assign(name);
  }} else return;
  e.preventDefault();
}});

async function save(quit) {{
  const body = {{
    pairs: st.filter(r => !r.bogus).map(r => [r.actor, r.text]),
    aliases: document.getElementById("aliases").value,
    quit: quit,
  }};
  const res = await fetch("/save", {{method: "POST", body: JSON.stringify(body)}});
  const msg = await res.text();
  dirty = res.ok ? false : dirty;
  render();
  document.getElementById("status").innerHTML =
    res.ok ? `<span class="saved">${{esc(msg)}}</span>` : `<span class="error">${{esc(msg)}}</span>`;
  if (res.ok && quit) setTimeout(() => window.close(), 300);
}}
document.getElementById("save").onclick = e => save(e.shiftKey);
window.addEventListener("beforeunload", e => {{ if (dirty) e.preventDefault(); }});

render();
</script>
</body>
</html>
"""


def parse_alias_block(alias_text: str) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for line in alias_text.strip().split("\n"):
        if not (line := line.strip()) or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"malformed alias line: {line!r}")
        canonical, alts = line.split("=", 1)
        aliases[canonical.strip()] = [
            alt for a in alts.split("|") if (alt := a.strip())
        ]
    return aliases


def run_server(text: str, payload: dict, out_path: Path, port: int) -> None:
    page = PAGE_TEMPLATE.format(
        title=out_path.name,
        payload=json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"),
        unattributed=json.dumps(UNATTRIBUTED),
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _respond(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._respond(200, page, "text/html; charset=utf-8")
            else:
                self._respond(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/save":
                self._respond(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            try:
                aliases = parse_alias_block(data.get("aliases", ""))
            except ValueError as e:
                self._respond(400, str(e).encode("utf-8"), "text/plain; charset=utf-8")
                return
            content = serialize_corpus_file(
                text, [tuple(p) for p in data["pairs"]], aliases
            )
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(out_path)
            message = f"сохранено: {out_path}"
            self._respond(200, message.encode("utf-8"), "text/plain; charset=utf-8")
            if data.get("quit"):
                threading.Thread(target=server.shutdown, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Annotating -> {out_path}\nOpen {url} (Ctrl+C to stop)")
    threading.Timer(0.3, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def run(cc, text_file: Path, out_path: Path, port: int = 8765) -> None:
    content = text_file.read_text(encoding="utf-8")
    prefill: dict[str, str] | None = None
    if DELIMITER in content:
        existing = parse_corpus_content(content, text_file)
        text = existing.text
        prefill = {replica: actor for actor, replica in existing.pairs}
    else:
        text = content.strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(cc, text, prefill)
    if not payload["replicas"]:
        print("No replicas extracted from the text — nothing to annotate.")
        return
    run_server(text, payload, out_path, port)
