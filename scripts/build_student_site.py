"""Build dist-student/: the site teams see.

Left out: host-trained weights, the results page, host-only text, and the code that runs the improved
model and the physics reference. Deploy this folder for students. Use app/ (host view) for the host session.
"""
from pathlib import Path
import re, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'dist-student'
PAGES=['index.html','how-it-works.html','team-kit.html']
ASSETS=['model.json','metadata.json']                       # the original network only
DOCS=['TEAM_BRIEF.md','EXPERIMENTS.md','TEAM_BRIEF.pdf','EXPERIMENTS.pdf','team-notebook.ipynb']
FORBIDDEN_FILES=['results.html','local-model','local-metadata','improved-model','neural-benchmark-summary','calibration.json']
# Must not appear anywhere in the student pages or scripts (checked case-insensitively).
FORBIDDEN_TEXT=['improved','calibrat','localneural','local-neural','skipweight','wallweight','structuredstep',
                'contact-repair','gated','host-only','host=','sessionstorage','inferred','known-geometry']

STUDENT_MODE_JS="document.documentElement.dataset.mode='student';\n"
STUDENT_INFO_JS='''const $=id=>document.getElementById(id);
async function asset(name){const r=await fetch(`../assets/${name}.json`);if(!r.ok)throw Error(`Could not load ${name}.json`);return r.json();}
try{
  if($('training-details')){
    const meta=await asset('metadata');
    $('training-details').textContent=`Original network: ${meta.trainingSamples.toLocaleString()} transitions, initial speeds 0.008–0.040. All distances are percentages of table width; target radius is 6.5%.`;
  }
}catch(error){const box=$('error');if(box){box.hidden=false;box.textContent=error.message||String(error);}}
'''

def strip_host_markup(html):
    for tag in ('a','span','section'):
        html=re.sub(rf'<{tag}\b[^>]*data-host-only[^>]*>.*?</{tag}>','',html,flags=re.S)
    html=re.sub(r'<option[^>]*value="(improved-neural|inferred-neural|calibrated)"[^>]*>.*?</option>','',html,flags=re.S)
    return html

def drop_function(src,name):
    """Remove a top-level function (optionally exported or async) whose closing brace is at column 0."""
    m=re.search(rf'^(?:export )?(?:async )?function {name}\(.*?^\}}\n',src,flags=re.S|re.M)
    if not m:sys.exit(f'Could not find function {name} to remove; update this script.')
    return src[:m.start()]+src[m.end():]

def replace_once(src,old,new,label):
    if src.count(old)!=1:sys.exit(f'Student transform failed ({label}): the source changed, update this script.')
    return src.replace(old,new)

def student_engine(src):
    for name in ('validateLocalModel','localNeuralStep','validateInferredModel','inferredStep','validateImprovedModel','improvedStep','validateCalibration','structuredStep'):
        src=drop_function(src,name)
    src=re.sub(r'^// The same model, routed with geometry[^\n]*\n','',src,count=1,flags=re.M)
    src,count=re.subn(r'^export const DEFAULT_GEOMETRY=[^\n]*\n','',src,count=1,flags=re.M)
    if count!=1:sys.exit('Student transform failed (DEFAULT_GEOMETRY): update this script.')
    src=replace_once(src,
        "  if(controller==='improved-neural')return improvedStep(s,improvedModel);\n  if(controller==='inferred-neural')return inferredStep(s,inferredModel);\n  if(!['contact-repair','calibrated'].includes(controller))throw Error('Unknown controller.');\n  return structuredStep(s,parameters,controller==='contact-repair'?m:null);\n",
        "  throw Error('Unknown controller.');\n",'controllerStep')
    src=replace_once(src,
        "  if(controller==='improved-neural')validateImprovedModel(improvedModel);\n  else if(controller==='inferred-neural')validateInferredModel(inferredModel);\n  else if(controller==='team')validateTeamModel(teamModel);\n  else if(controller!=='neural')validateCalibration(parameters);\n",
        "  if(controller==='team')validateTeamModel(teamModel);\n  else if(controller!=='neural')throw Error('Unknown controller.');\n",'plan')
    return src

def student_app(src):
    src=replace_once(src,"validateImprovedModel,validateInferredModel,validateTeamModel,","validateTeamModel,",'import 1')
    src=replace_once(src,"controllerStep,validateCalibration,","controllerStep,",'import 2')
    src=replace_once(src,"const hostMode=document.documentElement?.dataset?.mode==='host';\n","",'host flag')
    src=re.sub(r"// Host view only:[^\n]*\n(// [^\n]*\n)*","",src)
    src=drop_function(src,'loadHostAssets')
    src=replace_once(src,"if(!(hostMode&&await loadHostAssets()))useStudentView();","useStudentView();",'load')
    src=replace_once(src,
        "const controllerNames={'neural':'Original NN','improved-neural':'Known-geometry model','inferred-neural':'New approach (inferred geometry)','previous-neural':'Previous improved NN','team':'Your team NN','contact-repair':'Contact repair','calibrated':'Calibrated reference'};",
        "const controllerNames={'neural':'Original NN','team':'Your team NN'};",'names')
    for pattern in (r"'improved-neural':'[^']*',",r"'inferred-neural':'[^']*',",r"'contact-repair':'[^']*',",r",'calibrated':'[^']*'"):
        src,count=re.subn(pattern,'',src)
        if count!=1:sys.exit(f'Student transform failed (controller note {pattern}): update this script.')
    src=re.sub(r"function useStudentView\(\)\{[^\n]*\}\n","function useStudentView(){}\n",src,count=1)
    return src

def student_worker(src):
    src=replace_once(src,"import {plan,validateModel,validateCalibration,validateImprovedModel,validateInferredModel,validateTeamModel}","import {plan,validateModel,validateTeamModel}",'worker import')
    return replace_once(src,
        "parameters=event.data.parameters?validateCalibration(event.data.parameters):null;improvedModel=event.data.improvedModel?validateImprovedModel(event.data.improvedModel):null;inferredModel=event.data.inferredModel?validateInferredModel(event.data.inferredModel):null;",
        "parameters=null;improvedModel=null;inferredModel=null;",'worker line')

def write(path,text):
    path.write_text(text,encoding='utf8',newline='\n')

def main():
    # Clear old files. Windows or OneDrive can refuse to remove a folder, so keep going and let the copies overwrite.
    shutil.rmtree(OUT,ignore_errors=True)
    for folder in ('assets','student','packages'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    for name in PAGES:write(OUT/name,strip_host_markup((ROOT/'app'/name).read_text(encoding='utf8')))
    shutil.copy(ROOT/'app/style.css',OUT/'style.css')
    for name,transform in (('engine.mjs',student_engine),('app.mjs',student_app),('worker.mjs',student_worker)):
        # improvedModel is only an unused pass-through argument here, so give it a neutral name.
        write(OUT/name,transform((ROOT/'app'/name).read_text(encoding='utf8')).replace('improvedModel','extraModel').replace('inferredModel','extraModel2'))
    write(OUT/'mode.js',STUDENT_MODE_JS);write(OUT/'info.mjs',STUDENT_INFO_JS)
    for name in ASSETS:shutil.copy(ROOT/'assets'/name,OUT/'assets'/name)
    for name in DOCS:shutil.copy(ROOT/'student'/name,OUT/'student'/name)
    shutil.copy(ROOT/'packages/student-kit.zip',OUT/'packages/student-kit.zip')

    # Verify. The student site must not contain or name host material.
    problems=[]
    for name in PAGES+['app.mjs','engine.mjs','worker.mjs','info.mjs','mode.js']:
        text=(OUT/name).read_text(encoding='utf8').lower()
        problems+=[f'{name} still contains "{w}"' for w in FORBIDDEN_TEXT if w in text]
        problems+=[f'{name} names {w}' for w in FORBIDDEN_FILES if w in text]
    problems+=[f'host asset present: {p.name}' for p in (OUT/'assets').iterdir() if p.name not in ASSETS]
    if (OUT/'results.html').exists():problems.append('results.html was copied')
    if shutil.which('node'):
        for name in ('engine.mjs','app.mjs','worker.mjs','info.mjs'):
            check=subprocess.run(['node','--input-type=module','--check'],input=(OUT/name).read_bytes(),capture_output=True)
            if check.returncode:problems.append(f'{name} has a syntax error after stripping: {check.stderr.decode()[:200]}')
    if problems:sys.exit('Student site check failed:\n  '+'\n  '.join(problems))
    print('Built',OUT.relative_to(ROOT),'with',sum(1 for p in OUT.rglob('*') if p.is_file()),'files.')
    print('No host weights, results page, improved-model code or physics-reference code is included.')

if __name__=='__main__':main()
