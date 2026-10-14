/* Student view is the default. Host view (?host=1) adds the trained reference models and host-only pages.
   This only changes what the page shows. The student site build (scripts/build_student_site.py) also leaves the host weights out. */
(function(){
  var mode='student',root=document.documentElement;
  try{
    var q=new URLSearchParams(location.search).get('host');
    if(q==='1')sessionStorage.setItem('puckHost','1');
    if(q==='0')sessionStorage.removeItem('puckHost');
    if(sessionStorage.getItem('puckHost')==='1')mode='host';
  }catch(e){mode=new URLSearchParams(location.search).get('host')==='1'?'host':'student';}
  root.dataset.mode=mode;
  if(mode!=='host'&&root.hasAttribute('data-host-page'))location.replace('index.html');
})();
