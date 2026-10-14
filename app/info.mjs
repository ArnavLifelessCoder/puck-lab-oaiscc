const $=id=>document.getElementById(id),host=document.documentElement.dataset.mode==='host';
const unit=n=>`${(n*100).toFixed(1)}%`;
const names={'neural':'Original NN','previous-neural':'Previous improved NN','improved-neural':'Known-geometry model'};
async function asset(name){const r=await fetch(`../assets/${name}.json`);if(!r.ok)throw Error(`Could not load ${name}.json`);return r.json();}
function fail(error){const box=$('error');if(box){box.hidden=false;box.textContent=error.message||String(error);}}
async function results(){
  const comparison=await asset('neural-benchmark-summary');
  const keys=['neural','previous-neural','improved-neural'];
  let extra='';
  try{
    const inf=await asset('inferred-results');
    const bar=(h,t)=>`<td class="bar-cell"><div class="bar" role="img" aria-label="${h} of ${t} targets reached"><i style="width:${(100*h/t).toFixed(1)}%"></i></div></td>`;
    for(const [label,r] of [['New approach: inferred geometry, larger data',inf.results.expanded.held_out],['New approach: inferred geometry, original data only',inf.results.original.held_out]])
      extra+=`<tr><td>${label}</td>${bar(r.hits,r.total)}<td>${r.hits} / ${r.total}</td><td>${unit(r.mean_miss)}</td><td>n/a</td></tr>`;
    const g=inf.inferred_geometry,t=inf.true_geometry,rows=[['Wall, low side','wall_lo'],['Wall, high side','wall_hi'],['Bumper centre, x','cx'],['Bumper centre, y','cy'],['Bumper radius','radius'],['Drag per step','drag']];
    if($('geometry-rows'))$('geometry-rows').innerHTML=rows.map(([name,k])=>`<tr><td>${name}</td><td>${g[k].toFixed(4)}</td><td>${t[k].toFixed(4)}</td></tr>`).join('');
  }catch(e){/* the new-approach assets are optional */}
  $('benchmark-results').innerHTML=keys.map(k=>{const r=comparison.summary[k],pct=(100*r.hits/r.total).toFixed(1);return `<tr><td>${names[k]}</td><td class="bar-cell"><div class="bar" role="img" aria-label="${r.hits} of ${r.total} targets reached"><i style="width:${pct}%"></i></div></td><td>${r.hits} / ${r.total}</td><td>${unit(r.mean_miss)}</td><td>${r.regressed.length}</td></tr>`;})+extra;
}
async function training(){
  const meta=await asset('metadata'),base=`Original network: ${meta.trainingSamples.toLocaleString()} transitions, initial speeds 0.008–0.040.`,end=' All distances are percentages of table width; target radius is 6.5%.';
  if(!host){$('training-details').textContent=base+end;return;}
  const improved=await asset('local-metadata');
  $('training-details').textContent=`${base} The improved models share ${improved.training_samples.toLocaleString()} training observations covering speeds up to 0.115, with ${improved.validation_samples.toLocaleString()} separate validation states. The newest model changes the coordinate representation and neural heads.${end}`;
}
try{
  if(host&&$('benchmark-results'))await results();
  if($('training-details'))await training();
}catch(error){fail(error);}
