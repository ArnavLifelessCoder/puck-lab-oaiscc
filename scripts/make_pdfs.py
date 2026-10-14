"""Turn the Markdown guides into PDFs for people without a Markdown viewer.

Needs: pip install markdown, and Microsoft Edge or Google Chrome (used headless to print the PDF).
Run from anywhere:  python scripts/make_pdfs.py
"""
from pathlib import Path
import html, re, shutil, subprocess, sys, tempfile
try:
    import markdown
except ImportError:
    sys.exit('Install the converter first:  python -m pip install markdown')
ROOT=Path(__file__).resolve().parents[1]
GUIDES=[  # (markdown file, pdf file, document title)
    ('host/PROJECTED_WORKSHOP_RUNBOOK.md','host/PROJECTED_WORKSHOP_RUNBOOK.pdf','Puck Lab: projected workshop runbook'),
    ('student/TEAM_BRIEF.md','student/TEAM_BRIEF.pdf','Puck Lab: team brief'),
    ('student/EXPERIMENTS.md','student/EXPERIMENTS.pdf','Puck Lab: experiment cards'),
]
BROWSERS=[r'C:\Program Files\Google\Chrome\Application\chrome.exe',r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
          r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',r'C:\Program Files\Microsoft\Edge\Application\msedge.exe']
CSS='''
@page{size:A4;margin:16mm 15mm 18mm;@bottom-center{content:counter(page) " / " counter(pages);font:9pt "Segoe UI",Arial,sans-serif;color:#64748b}}
*{box-sizing:border-box}
body{margin:0;color:#1e293b;font:10pt/1.5 "Segoe UI",Arial,sans-serif}
h1{margin:0 0 4pt;color:#0f172a;font-size:21pt;line-height:1.15}
h2{margin:18pt 0 6pt;padding-bottom:3pt;border-bottom:1.5px solid #c7d2fe;color:#1e3a8a;font-size:14pt;break-after:avoid}
h3{margin:12pt 0 4pt;color:#0f172a;font-size:11pt;break-after:avoid}
p{margin:0 0 7pt;orphans:3;widows:3}
p:has(+pre),p:has(+table),p:has(+ul),p:has(+ol){break-after:avoid}
a{color:#1d4ed8;text-decoration:none}
blockquote{margin:8pt 0;padding:7pt 12pt;border-left:3px solid #7c3aed;background:#f5f3ff;color:#312e81;break-inside:avoid}
blockquote p{margin:0}
code{font:8.6pt Consolas,"Cascadia Code",monospace;padding:1pt 3pt;border-radius:3pt;background:#eef2ff;color:#1e3a8a}
pre{margin:8pt 0;padding:8pt 10pt;border:1px solid #cbd5e1;border-radius:6pt;background:#f8fafc;color:#0f172a;white-space:pre-wrap;word-break:break-word;break-inside:avoid}
pre code{padding:0;background:none;color:inherit;font-size:8.3pt;line-height:1.4}
table{width:100%;margin:8pt 0;border-collapse:collapse;font-size:8.8pt;break-inside:auto}
th{padding:4pt 6pt;background:#eef2ff;color:#1e3a8a;text-align:left;border:1px solid #cbd5e1}
td{padding:4pt 6pt;border:1px solid #cbd5e1;vertical-align:top}
tr{break-inside:avoid}
ul,ol{margin:0 0 8pt;padding-left:18pt}li{margin:2pt 0}
hr{border:0;border-top:1px solid #cbd5e1;margin:12pt 0}
.meta{margin:0 0 12pt;color:#64748b;font-size:8.5pt}
'''

def browsers():
    found=[path for path in BROWSERS if Path(path).is_file()]
    if not found:sys.exit('Chrome or Edge was not found. Install one, or open the Markdown in any editor.')
    return found

def page(markdown_text,title):
    body=markdown.markdown(markdown_text,extensions=['tables','fenced_code','sane_lists'])
    body=re.sub(r'<li>\[ \]\s*','<li>&#9744; ',body);body=re.sub(r'<li>\[[xX]\]\s*','<li>&#9745; ',body)
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>'

def print_pdf(page_file,out):
    """Try each installed browser until one writes a PDF (a browser already running can swallow the request)."""
    for exe in browsers():
        with tempfile.TemporaryDirectory() as profile:
            out.unlink(missing_ok=True)
            subprocess.run([exe,'--headless=new','--disable-gpu','--no-pdf-header-footer',f'--user-data-dir={profile}',f'--print-to-pdf={out}',page_file.as_uri()],
                           capture_output=True,timeout=120)
            if out.is_file() and out.stat().st_size>2000:return True
    return False

def main():
    for source,target,title in GUIDES:
        src,dst=ROOT/source,ROOT/target
        if not src.is_file():sys.exit(f'Missing {source}')
        with tempfile.TemporaryDirectory() as work:
            page_file=Path(work)/'guide.html';page_file.write_text(page(src.read_text(encoding='utf8'),title),encoding='utf8')
            out=Path(work)/'guide.pdf'
            if not print_pdf(page_file,out):sys.exit(f'No browser produced a PDF for {source}.')
            shutil.copy(out,dst)
        print('Wrote',target)

if __name__=='__main__':main()
