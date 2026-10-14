"""Build dist-host/: the full host site (both models, results, pointers), with host view always on.

This is the HOST version. It contains the trained reference models, so keep it private or owner-only.
Deploy the contents of dist-host/ at the site root. The student site is built separately (build_student_site.py).
"""
from pathlib import Path
import re, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'dist-host'
PAGES=['index.html','results.html','how-it-works.html','team-kit.html']
CODE=['style.css','app.mjs','engine.mjs','worker.mjs','info.mjs']
ASSETS=['model.json','metadata.json','calibration.json','neural-benchmark-summary.json','local-model.json','local-metadata.json','inferred-model.json','inferred-results.json']
DOCS=['TEAM_BRIEF.md','EXPERIMENTS.md','TEAM_BRIEF.pdf','EXPERIMENTS.pdf','team-notebook.ipynb']
HOST_MODE_JS="document.documentElement.dataset.mode='host';\n"

def main():
    shutil.rmtree(OUT,ignore_errors=True)
    for folder in ('assets','student','packages'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    for name in PAGES:
        html=(ROOT/'app'/name).read_text(encoding='utf8')
        html=re.sub(r'<a class="host-chip"[^>]*>.*?</a>','',html,flags=re.S)        # no "exit host view" link: this site is the host site
        (OUT/name).write_text(html,encoding='utf8',newline='\n')
    for name in CODE:shutil.copy(ROOT/'app'/name,OUT/name)
    (OUT/'mode.js').write_text(HOST_MODE_JS,encoding='utf8',newline='\n')
    for name in ASSETS:shutil.copy(ROOT/'assets'/name,OUT/'assets'/name)
    for name in DOCS:shutil.copy(ROOT/'student'/name,OUT/'student'/name)
    shutil.copy(ROOT/'packages/student-kit.zip',OUT/'packages/student-kit.zip')
    problems=[]
    for name in ASSETS:
        if not (OUT/'assets'/name).stat().st_size:problems.append(f'{name} is empty')
    if shutil.which('node'):
        for name in ('engine.mjs','app.mjs','worker.mjs','info.mjs'):
            if subprocess.run(['node','--input-type=module','--check'],input=(OUT/name).read_bytes(),capture_output=True).returncode:problems.append(f'{name} has a syntax error')
    if problems:sys.exit('Host site check failed:\n  '+'\n  '.join(problems))
    print('Built',OUT.relative_to(ROOT),'with',sum(1 for p in OUT.rglob('*') if p.is_file()),'files. HOST site: contains the reference models, so keep it private.')

if __name__=='__main__':main()
