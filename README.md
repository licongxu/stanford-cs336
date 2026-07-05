# Stanford CS336 study workspace

Local workspace for [CS336: Language Modeling from Scratch](https://github.com/stanford-cs336/lectures).

## Layout

- `cs336_notes/` — Obsidian study notes
- `lectures/` — course lecture materials (executable lectures)
- `LLM-Everything/` — supplementary LLM reference notes
- `cs336_assignment/` — assignment solutions / work
- Run lecture commands from inside `lectures/`

## Setup

```bash
cd lectures
uv sync
git clone https://github.com/percyliang/edtrace
cd edtrace/frontend && npm install
```

## Compile and view a lecture

```bash
cd lectures
python -m edtrace.execute -m lecture_01
npm run --prefix=edtrace/frontend dev
```

Open http://localhost:5173/?trace=lecture_01
