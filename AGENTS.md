# Agent rules

## Context economy
- Read only the files I mention. Do not explore the repo unless asked.
- If you need a global view, read README.md only.
- Never open image files unless I explicitly ask.
- Do not list or open directories I did not mention. If you think one is needed, ask first and say why.
- Skip large data files (csv, parquet, logs) and generated or dependency folders (.venv, node_modules, outputs).
- Prefer grep and targeted line ranges over reading whole files. Never re-read a file already in context.
- Edit in place; do not reprint unchanged code.
- Keep replies short: no preamble, no recap, no unsolicited suggestions.
- Limit long command output (e.g. `| tail -n 30`).

## Python style
- Everything in English: code, comments, docstrings, print and log messages.
- Every module starts with a concise module docstring.
- Function docstrings: one concise line.
- Comments: very concise, no step numbers. In longer modules, separate main parts and sub-parts:
  - Level 1: `# %% UPPERCASE TITLE` (e.g. `# %% FUNCTIONS`)
  - Level 2: `# UPPERCASE TITLE` (e.g. `# MATCHING FUNCTIONS`)
  - Free comments: normal case (e.g. `# Disable notifications.`)