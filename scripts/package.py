"""Build portable archives that preserve the organized project structure."""
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
PACKAGES=ROOT/'packages'
PACKAGES.mkdir(exist_ok=True)
student=['student/TEAM_BRIEF.md','student/EXPERIMENTS.md','student/TEAM_BRIEF.pdf','student/EXPERIMENTS.pdf','student/run_experiment.py','student/team-notebook.ipynb','student/lab.py','student/repair.py','student/check_submission.py','student/requirements.txt','student/training-requirements.txt','student/train_repair.py','student/neural_model.py','assets/neural-training.npz','assets/model.json','assets/metadata.json','assets/parity.json','assets/training-observations.npz','assets/development-scenes.json']
readme="""# Student package

Keep assets/ and student/ together. Install student/training-requirements.txt before the event. Open a terminal in student/, then run python -m jupyterlab team-notebook.ipynb. Train/edit the neural network with student/train_repair.py, then edit repair.py and run python check_submission.py from student/. Generated reports go into student/results/. One laptop per team; no paid account or API key. Follow student/TEAM_BRIEF.md. Host solutions and confirmation scenes are excluded.
"""
with zipfile.ZipFile(PACKAGES/'student-kit.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in student:archive.write(ROOT/name,'puck-lab/'+name)
    archive.writestr('puck-lab/README.md',readme)
with zipfile.ZipFile(PACKAGES/'host-kit.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in ROOT.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if any(part in ['.git','__pycache__','.ipynb_checkpoints','verification','packages','deployment','dist-student','dist-host','scratch'] for part in relative.parts):continue
        if relative.parts[:2]==('student','results'):continue
        if path.suffix=='.zip':continue
        archive.write(path,'puck-lab/'+str(relative))
with zipfile.ZipFile(PACKAGES/'student-kit.zip') as archive:
    names=archive.namelist()
    assert not any('confirmation' in n or '/host/' in n or 'calibration' in n or 'controllers.py' in n for n in names)
    assert archive.testzip() is None
with zipfile.ZipFile(PACKAGES/'host-kit.zip') as archive:assert archive.testzip() is None
print('Rebuilt packages/student-kit.zip and packages/host-kit.zip; archive integrity and student allowlist passed.')
