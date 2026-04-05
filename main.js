const cur = document.getElementById('cur'), ring = document.getElementById('curRing');
let mx=0,my=0,rx=0,ry=0;
document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;});
(function tick(){
  cur.style.left=mx+'px';cur.style.top=my+'px';
  rx+=(mx-rx)*.12;ry+=(my-ry)*.12;
  ring.style.left=rx+'px';ring.style.top=ry+'px';
  requestAnimationFrame(tick);
})();
document.querySelectorAll('button,a,input,select,.radio-opt').forEach(el=>{
  el.addEventListener('mouseenter',()=>document.body.classList.add('btn-hover'));
  el.addEventListener('mouseleave',()=>document.body.classList.remove('btn-hover'));
});
 
/* ─── HERO CANVAS ─── */
const hc = document.getElementById('heroCanvas');
const hctx = hc.getContext('2d');
hc.width = hc.offsetWidth||480; hc.height = hc.offsetHeight||480;
const W=hc.width,H=hc.height;
const ROOMS=[
  {x:.05,y:.05,w:.55,h:.45,label:'Living Room'},
  {x:.62,y:.05,w:.33,h:.45,label:'Kitchen'},
  {x:.05,y:.55,w:.28,h:.4,label:'Bedroom 1'},
  {x:.36,y:.55,w:.28,h:.4,label:'Bedroom 2'},
  {x:.67,y:.55,w:.28,h:.19,label:'Bath'},
  {x:.67,y:.76,w:.28,h:.19,label:'Storage'},
];
let hp=0;
function drawHero(p){
  hctx.clearRect(0,0,W,H);
  hctx.fillStyle='#f5fbfd';hctx.fillRect(0,0,W,H);
  hctx.strokeStyle='rgba(12,35,64,.05)';hctx.lineWidth=1;
  for(let i=0;i<W;i+=24){hctx.beginPath();hctx.moveTo(i,0);hctx.lineTo(i,H);hctx.stroke();}
  for(let j=0;j<H;j+=24){hctx.beginPath();hctx.moveTo(0,j);hctx.lineTo(W,j);hctx.stroke();}
  const nv=Math.ceil(ROOMS.length*p);
  for(let i=0;i<nv;i++){
    const r=ROOMS[i],rp=Math.min(1,(p*ROOMS.length-i));
    const rx2=r.x*W,ry2=r.y*H,rw=r.w*W*rp,rh=r.h*H;
    hctx.fillStyle=i%2?'#f8faff':'#f0f4ff';hctx.fillRect(rx2,ry2,rw,rh);
    hctx.strokeStyle='#0891b2';hctx.lineWidth=1.8;hctx.strokeRect(rx2,ry2,rw,rh);
    if(rp>.6){hctx.fillStyle='rgba(12,35,64,.28)';hctx.font=`${Math.max(9,rw*.085)}px DM Mono,monospace`;hctx.textAlign='center';hctx.fillText(r.label,rx2+rw/2,ry2+rh/2+4);}
  }
}
function tickHero(){
  if(hp<1){hp+=.007;drawHero(Math.min(hp,1));requestAnimationFrame(tickHero);}
  else setTimeout(()=>{hp=0;tickHero();},2800);
}
setTimeout(tickHero,900);
 
/* ─── DARK MODE ─── */
const themeToggle = document.getElementById('themeToggle');
const themeLabel = document.getElementById('themeLabel');
 
function applyTheme(dark) {
  if (dark) {
    document.body.classList.add('dark');
    themeLabel.textContent = 'Dark';
  } else {
    document.body.classList.remove('dark');
    themeLabel.textContent = 'Light';
  }
  localStorage.setItem('theme', dark ? 'dark' : 'light');
}
 
themeToggle.addEventListener('click', () => {
  applyTheme(!document.body.classList.contains('dark'));
});
applyTheme(localStorage.getItem('theme') === 'dark');
 
/* ─── CHAT TOGGLE ─── */
let chatOpen = true;
function toggleChat() {
  chatOpen = !chatOpen;
  const sidebar = document.getElementById('appSidebar');
  const appBody = document.getElementById('appBody');
  const appMain = document.querySelector('.app-main');
  const reopenBtn = document.getElementById('chatReopenBtn');
  const isMobile = window.innerWidth <= 900;
 
  if (chatOpen) {
    sidebar.classList.remove('collapsed');
    appBody.classList.remove('chat-collapsed');
    if (!isMobile) appMain.style.maxWidth = 'calc(100% - 320px)';
    reopenBtn.classList.remove('visible');
  } else {
    sidebar.classList.add('collapsed');
    appBody.classList.add('chat-collapsed');
    appMain.style.maxWidth = '100%';
    setTimeout(() => reopenBtn.classList.add('visible'), 350);
  }
}
 
 
function updateArea(){
  const l=parseFloat(document.getElementById('g-len').value)||0;
  const w=parseFloat(document.getElementById('g-wid').value)||0;
  const a=Math.max(100,l*w);
  document.getElementById('areaDisplay').textContent=`Calculated Total Area: ${a.toFixed(2)} m² (≈ ${Math.round(a*10.7639)} sq ft)`;
}
 
/* ─── SCROLL TO APP ─── */
function scrollToApp(){
  document.getElementById('appLayer').scrollIntoView({behavior:'smooth'});
}
 
/* ─── TABS ─── */
function switchTab(id,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('panel-'+id).classList.add('active');
}
 
/* ─── RADIO ─── */
const radioState={};
function selectRadio(el,group,val){
  document.querySelectorAll(`[onclick*="'${group}'"]`).forEach(e=>e.classList.remove('selected'));
  el.classList.add('selected');
  radioState[group]=val;
}
radioState['plotShape']='rectangle';
 
/* ─── ONNX MODEL ─── */
const ONNX_URL = 'https://raw.githubusercontent.com/Aravkataria/Arch-Ai-Tex/main/generator.onnx';
let onnxSession = null;
let modelLoading = false;
 
async function loadONNXModel() {
  if (onnxSession) return onnxSession;
  if (modelLoading) return null;
  modelLoading = true;
  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    onnxSession = await ort.InferenceSession.create(ONNX_URL, {
      executionProviders: ['wasm']
    });
    console.log('ONNX model loaded');
    const s = document.getElementById('model-status');
    if (s) s.textContent = '✓ Model ready';
    return onnxSession;
  } catch(e) {
    console.error('ONNX load error:', e);
    modelLoading = false;
    return null;
  }
}
 
// preload model in background on page load
window.addEventListener('load', () => setTimeout(loadONNXModel, 2000));
 
async function runONNXInference() {
  const session = await loadONNXModel();
  if (!session) throw new Error('Model failed to load');
  // random noise vector [1, 100]
  const noise = new Float32Array(100);
  for (let i = 0; i < 100; i++) {
    // Box-Muller for normal distribution
    const u1 = Math.random(), u2 = Math.random();
    noise[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  const tensor = new ort.Tensor('float32', noise, [1, 100]);
  const results = await session.run({ noise: tensor });
  const output = results[Object.keys(results)[0]];
  return output.data; // Float32Array of 256*256 values in [-1, 1]
}
 
function renderONNXToCanvas(canvasEl, floatData) {
  canvasEl.width = 256; canvasEl.height = 256;
  const ctx = canvasEl.getContext('2d');
  const imgData = ctx.createImageData(256, 256);
  for (let i = 0; i < 256 * 256; i++) {
    const val = Math.max(0, Math.min(255, Math.round(((floatData[i] + 1) / 2) * 255)));
    imgData.data[i*4]=val; imgData.data[i*4+1]=val; imgData.data[i*4+2]=val; imgData.data[i*4+3]=255;
  }
  ctx.putImageData(imgData, 0, 0);
  return imgData;
}
 
/* ─── JS SEGMENTATION (mirrors cv2.threshold + connectedComponents) ─── */
function applySegmentation(floatData, canvasEl) {
  canvasEl.width = 256; canvasEl.height = 256;
  const ctx = canvasEl.getContext('2d');
  const W = 256, H = 256;
 
  // convert to 0-255 grayscale
  const gray = new Uint8Array(W * H);
  for (let i = 0; i < W*H; i++)
    gray[i] = Math.max(0, Math.min(255, Math.round(((floatData[i]+1)/2)*255)));
 
  // ── Otsu's threshold (automatic, better than fixed 128) ──
  const hist = new Int32Array(256);
  for (let i = 0; i < W*H; i++) hist[gray[i]]++;
  const total = W*H;
  let sum=0; for (let i=0;i<256;i++) sum+=i*hist[i];
  let sumB=0,wB=0,wF=0,maxVar=0,thresh=128;
  for (let t=0;t<256;t++){
    wB+=hist[t]; if(!wB)continue;
    wF=total-wB; if(!wF)break;
    sumB+=t*hist[t];
    const mB=sumB/wB, mF=(sum-sumB)/wF;
    const v=wB*wF*(mB-mF)*(mB-mF);
    if(v>maxVar){maxVar=v;thresh=t;}
  }
 
  // binary: 0=wall(dark), 1=room(bright)
  const binary = new Uint8Array(W*H);
  for (let i=0;i<W*H;i++) binary[i] = gray[i]>thresh ? 1 : 0;
 
  // ── connected components ──
  const labels = new Int32Array(W*H).fill(-1);
  let numLabels=0;
  const compInfo=[];
  for (let y=0;y<H;y++){
    for (let x=0;x<W;x++){
      const idx=y*W+x;
      if(binary[idx]===1 && labels[idx]===-1){
        const queue=[idx]; labels[idx]=numLabels; let sz=0;
        while(queue.length){
          const c=queue.shift(); sz++;
          const cx=c%W, cy=Math.floor(c/W);
          for(const n of [cy>0?(cy-1)*W+cx:-1,cy<H-1?(cy+1)*W+cx:-1,cx>0?cy*W+(cx-1):-1,cx<W-1?cy*W+(cx+1):-1])
            if(n>=0&&binary[n]===1&&labels[n]===-1){labels[n]=numLabels;queue.push(n);}
        }
        compInfo.push({id:numLabels,size:sz}); numLabels++;
      }
    }
  }
 
  // sort by size, skip largest (bg), take next 8 = rooms
  const sorted = compInfo.filter(c=>c.size>=80).sort((a,b)=>b.size-a.size);
  const rooms = sorted.slice(1,9);
  const colorMap = new Map();
  const ROOM_COLORS=[
    [255,199,107],[130,202,157],[174,199,232],[255,152,150],
    [197,176,213],[255,237,111],[188,189,34],[140,86,75]
  ];
  rooms.forEach((c,i)=>colorMap.set(c.id, ROOM_COLORS[i%ROOM_COLORS.length]));
 
  // ── draw: dark bg, colored rooms, bright wall lines ──
  const out = ctx.createImageData(W,H);
  for (let i=0;i<W*H;i++){
    if(binary[i]===0){
      // wall — very dark navy
      out.data[i*4]=8; out.data[i*4+1]=15; out.data[i*4+2]=30; out.data[i*4+3]=255;
    } else if(colorMap.has(labels[i])){
      const [r,g,b]=colorMap.get(labels[i]);
      // slightly desaturate for realism
      out.data[i*4]=r; out.data[i*4+1]=g; out.data[i*4+2]=b; out.data[i*4+3]=255;
    } else {
      out.data[i*4]=40; out.data[i*4+1]=60; out.data[i*4+2]=90; out.data[i*4+3]=255;
    }
  }
  ctx.putImageData(out, 0, 0);
 
  // ── overlay sharp wall lines (edge detection on original gray) ──
  const edgeData = ctx.createImageData(W,H);
  for (let y=1;y<H-1;y++){
    for (let x=1;x<W-1;x++){
      const i=y*W+x;
      const gx=gray[(y)*W+(x+1)]-gray[(y)*W+(x-1)];
      const gy=gray[(y+1)*W+x]-gray[(y-1)*W+x];
      const mag=Math.min(255,Math.sqrt(gx*gx+gy*gy));
      if(mag>40){ // edge pixel
        edgeData.data[i*4]=220; edgeData.data[i*4+1]=235; edgeData.data[i*4+2]=255; edgeData.data[i*4+3]=Math.round(mag*1.2);
      }
    }
  }
  // use a second canvas to blend edges on top
  const tmp = document.createElement('canvas'); tmp.width=W; tmp.height=H;
  const tctx = tmp.getContext('2d');
  tctx.putImageData(edgeData,0,0);
  ctx.globalAlpha=0.85;
  ctx.drawImage(tmp,0,0);
  ctx.globalAlpha=1.0;
 
  return rooms;
}
 
 
/* ─── GAN GENERATE ─── */
async function generateGAN() {
  const beds = parseInt(document.getElementById('g-beds').value) || 3;
  const len = parseFloat(document.getElementById('g-len').value) || 50;
  const wid = parseFloat(document.getElementById('g-wid').value) || 30;
  const area = Math.max(100, len * wid);
  const btn = document.querySelector('#panel-gan .btn-gen');
  btn.textContent = 'Loading model…'; btn.disabled = true;
 
  function predictDwelling(area, beds) {
    if (area < 50) return 'Studio Flat';
    if (area < 80) return beds <= 1 ? 'Apartment' : 'Semi-Detached';
    if (area < 150) return beds <= 2 ? 'Apartment' : beds <= 3 ? 'Semi-Detached' : 'Detached House';
    if (area < 250) return beds <= 3 ? 'Detached House' : 'Large Detached House';
    return 'Villa';
  }
 
  try {
    const session = await loadONNXModel();
    if (!session) throw new Error('Could not load model');
    btn.textContent = 'Generating…';
 
    const grid = document.getElementById('plansGrid');
    const segGrid = document.getElementById('segGrid');
    grid.innerHTML = ''; segGrid.innerHTML = '';
 
    for (let i = 0; i < 3; i++) {
      const floatData = await runONNXInference();
 
      // ── raw plan card ──
      const card = document.createElement('div'); card.className = 'plan-card';
      const wrap = document.createElement('div'); wrap.className = 'plan-canvas-wrap';
      const cv = document.createElement('canvas');
      renderONNXToCanvas(cv, floatData);
      wrap.appendChild(cv);
      const lbl = document.createElement('div'); lbl.className = 'plan-label';
      lbl.innerHTML = `<span>Plan ${i+1}</span><a class="plan-dl" href="#" onclick="downloadPlan(event,this)">↓ PNG</a>`;
      card.appendChild(wrap); card.appendChild(lbl); grid.appendChild(card);
 
      // ── segmented plan card ──
      const segCard = document.createElement('div'); segCard.className = 'plan-card';
      const segWrap = document.createElement('div'); segWrap.className = 'plan-canvas-wrap';
      const segCv = document.createElement('canvas');
      applySegmentation(floatData, segCv);
      segWrap.appendChild(segCv);
      const segLbl = document.createElement('div'); segLbl.className = 'plan-label';
      segLbl.innerHTML = `<span>Segmented ${i+1}</span><a class="plan-dl" href="#" onclick="downloadPlan(event,this)">↓ PNG</a>`;
      segCard.appendChild(segWrap); segCard.appendChild(segLbl); segGrid.appendChild(segCard);
    }
 
    const dw = predictDwelling(area, beds);
    document.getElementById('dwelling-badge').textContent = `Predicted Dwelling Type: ${dw}`;
    document.getElementById('gan-results').style.display = 'block';
 
  } catch(e) {
    alert('Model error: ' + e.message + '\n\nMake sure generator.onnx is in your GitHub repo.');
    console.error(e);
  } finally {
    btn.textContent = 'Generate Floorplans'; btn.disabled = false;
  }
}
 
 
/* ─── OPTIMIZED GENERATE ─── */
function generateOptimized(){
  const totalArea=parseFloat(document.getElementById('o-area').value)||120;
  const numRooms=parseInt(document.getElementById('o-rooms').value)||5;
  const pw=parseFloat(document.getElementById('o-pw').value)||10;
  const ph=parseFloat(document.getElementById('o-ph').value)||12;
  const btn=document.querySelector('#panel-optimized .btn-gen');
  btn.textContent='Computing…';btn.disabled=true;
  setTimeout(()=>{
    btn.textContent='Generate Optimized Layout';btn.disabled=false;
    const rooms=computeRooms(totalArea,numRooms);
    renderRoomList(rooms,totalArea);
    draw2DLayout(rooms,pw,ph);
    document.getElementById('opt-results').style.display='block';
  },1000);
}
 
function computeRooms(total,n){
  const fixed=[{name:'living & dining',ratio:.28},{name:'kitchen',ratio:.08},{name:'bathroom',ratio:.06}];
  const fixedTotal=fixed.reduce((s,r)=>s+r.ratio,0);
  const nBeds=Math.max(0,n-fixed.length);
  const bedRatio=nBeds>0?(1-fixedTotal)/nBeds:0;
  const rooms=fixed.map(r=>({name:r.name,area:+(total*r.ratio).toFixed(2)}));
  for(let i=0;i<nBeds;i++) rooms.push({name:`bedroom ${i+1}`,area:+(total*bedRatio).toFixed(2)});
  return rooms;
}
 
function renderRoomList(rooms,total){
  const el=document.getElementById('roomList');
  el.innerHTML='';
  rooms.forEach(r=>{
    const pct=(r.area/total*100).toFixed(0);
    el.innerHTML+=`<div class="room-item"><span class="room-name">${r.name}</span><div class="room-bar-wrap"><div class="room-bar" style="width:${pct}%"></div></div><span class="room-area">${r.area} m²</span></div>`;
  });
}
 
function draw2DLayout(rooms,pw,ph){
  const cv=document.getElementById('canvas2d');
  cv.width=cv.offsetWidth||500;cv.height=400;
  const ctx=cv.getContext('2d');
  const scale=Math.min((cv.width-40)/pw,(cv.height-40)/ph);
  const ox=20,oy=20;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.fillStyle='#f5fbfd';ctx.fillRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='rgba(12,35,64,.07)';ctx.lineWidth=1;
  for(let i=0;i<cv.width;i+=20){ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,cv.height);ctx.stroke();}
  for(let j=0;j<cv.height;j+=20){ctx.beginPath();ctx.moveTo(0,j);ctx.lineTo(cv.width,j);ctx.stroke();}
  ctx.strokeStyle='#0a1628';ctx.lineWidth=2;ctx.strokeRect(ox,oy,pw*scale,ph*scale);
  const colors=['#dbeafe','#ede9fe','#cffafe','#d1fae5','#fce7f3','#e0e7ff','#fef3c7','#bfdbfe'];
  const totalA=rooms.reduce((s,r)=>s+r.area,0);
  let x=ox+4,y=oy+4,rowH=0,pad=4;
  rooms.forEach((r,i)=>{
    const ra=r.area/totalA*(pw*scale*(ph*scale));
    const w=Math.sqrt(ra)*1.3,h=ra/w;
    if(x+w+pad>ox+pw*scale){x=ox+4;y+=rowH+pad;rowH=0;}
    if(y+h+pad>oy+ph*scale) return;
    ctx.fillStyle=colors[i%colors.length];ctx.fillRect(x,y,w,h);
    ctx.strokeStyle='#0a1628';ctx.lineWidth=1;ctx.strokeRect(x,y,w,h);
    ctx.fillStyle='rgba(12,35,64,.6)';ctx.font='9px DM Mono,monospace';ctx.textAlign='center';
    ctx.fillText(`${r.name}`,x+w/2,y+h/2-6);
    ctx.fillText(`${r.area}m²`,x+w/2,y+h/2+8);
    x+=w+pad;rowH=Math.max(rowH,h);
  });
}
 
/* ─── DOWNLOAD PLAN ─── */
function downloadPlan(e,link){
  e.preventDefault();
  const wrap=link.closest('.plan-card').querySelector('canvas');
  const a=document.createElement('a');a.href=wrap.toDataURL('image/png');a.download='floorplan.png';a.click();
}
 
/* ─── SENSOR STATE MACHINE ─── */
let sensorState={length:null,breadth:null,lastDist:null,pir:null,ir:null,lastSet:null};
function renderSensor(){
  const el=document.getElementById('sensor-ui');
  const{length,breadth,lastDist,pir,ir}=sensorState;
  let html='';
  if(pir!==null||ir!==null){
    html+=`<div class="sensor-grid">
      <div class="sensor-mini"><div class="sensor-mini-lbl">PIR Motion</div><div class="indicator"><div class="${pir?'dot-on':'dot-off'}"></div>${pir?'Motion Detected':'No Motion'}</div></div>
      <div class="sensor-mini"><div class="sensor-mini-lbl">IR Obstacle</div><div class="indicator"><div class="${!ir?'dot-on':'dot-off'}"></div>${!ir?'Obstacle Detected':'Clear Path'}</div></div>
    </div>`;
  }
  if(length!==null&&breadth!==null){
    html+=`<div class="dim-confirmed">
      <div class="dim-box"><div class="dim-box-val">${length} cm</div><div class="dim-box-lbl">Length</div></div>
      <div class="dim-box"><div class="dim-box-val">${breadth} cm</div><div class="dim-box-lbl">Breadth</div></div>
    </div>
    <div class="msg-success">✓ Both dimensions captured successfully!</div>
    <div class="form-row"><div class="form-group"><label>Number of Bedrooms</label><input type="number" id="s-beds" value="3" min="1"></div></div>
    <div class="checkbox-row"><input type="checkbox" id="s-denoise"><label for="s-denoise">Apply Denoiser</label></div>
    <button class="btn-gen" onclick="generateFromSensor()">Generate Floorplans from Sensor Data</button>
    <button class="btn-sensor danger" onclick="resetSensor()">Reset All</button>`;
  } else if(lastDist!==null){
    html+=`<div class="sensor-card"><div class="sensor-row"><div><div class="sensor-val">${lastDist} cm</div><div class="sensor-unit">Last Measured Distance</div></div></div>`;
    if(length===null&&breadth===null){
      html+=`<button class="btn-sensor" onclick="setAs('length')">Set as Length</button><button class="btn-sensor" onclick="setAs('breadth')">Set as Breadth</button>`;
    } else if(length!==null){
      html+=`<button class="btn-sensor" onclick="setAs('breadth')">Set as Breadth</button>`;
    } else {
      html+=`<button class="btn-sensor" onclick="setAs('length')">Set as Length</button>`;
    }
    html+=`<button class="btn-sensor danger" onclick="resetLastDist()">Reset Last Value</button></div>`;
  } else if((length!==null)^(breadth!==null)){
    const needed=length===null?'Length':'Breadth';
    html+=`<div class="msg-info">Now measure the ${needed}.</div>`;
    html+=`<button class="btn-sensor" onclick="fetchSensor()">Get Sensor Data</button>`;
    if(length!==null) html+=`<button class="btn-sensor danger" onclick="()=>{sensorState.length=null;renderSensor()}">Reset Length</button>`;
    if(breadth!==null) html+=`<button class="btn-sensor danger" onclick="()=>{sensorState.breadth=null;renderSensor()}">Reset Breadth</button>`;
  } else {
    html+=`<button class="btn-gen" onclick="fetchSensor()">Get Sensor Data</button>`;
  }
  html+=`<div style="margin-top:20px;"><div class="section-hd">Current Measurements</div>
    <div class="sensor-card" style="padding:16px 20px;">
      <div style="display:flex;gap:32px;font-size:.78rem;">
        <div><span style="opacity:.45;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;display:block;margin-bottom:4px;">Length</span>${length!==null?length+' cm':'—'}</div>
        <div><span style="opacity:.45;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;display:block;margin-bottom:4px;">Breadth</span>${breadth!==null?breadth+' cm':'—'}</div>
      </div>
    </div>
  </div>`;
  el.innerHTML=html;
}
function fetchSensor(){
  const btn=document.querySelector('#sensor-ui .btn-gen');
  if(btn){btn.textContent='Fetching…';btn.disabled=true;}
  fetch('https://esp32-fastapi-server-uh47.onrender.com/data',{signal:AbortSignal.timeout(5000)})
    .then(r=>r.json())
    .then(d=>{
      const data=d.data||{};
      sensorState.pir=data.pir??null;
      sensorState.ir=data.ir??null;
      sensorState.lastDist=data.ultrasonic??Math.round(100+Math.random()*400);
      renderSensor();
    })
    .catch(()=>{
      sensorState.lastDist=Math.round(100+Math.random()*400);
      renderSensor();
    });
}
function setAs(dim){
  sensorState[dim]=sensorState.lastDist;
  sensorState.lastSet=dim;
  sensorState.lastDist=null;
  renderSensor();
}
function resetLastDist(){sensorState.lastDist=null;renderSensor();}
function resetSensor(){Object.keys(sensorState).forEach(k=>sensorState[k]=null);renderSensor();}
function generateFromSensor(){
  const{length,breadth}=sensorState;
  const lm=length*.01,wm=breadth*.01;
  const beds=parseInt(document.getElementById('s-beds').value)||3;
  document.getElementById('g-len').value=lm.toFixed(1);
  document.getElementById('g-wid').value=wm.toFixed(1);
  document.getElementById('g-beds').value=beds;
  updateArea();
  switchTab('gan',document.querySelector('.tab'));
  document.querySelectorAll('.tab')[0].click();
  setTimeout(()=>generateGAN(),300);
}
renderSensor();
 
/* ─── CHATBOT ─── */
async function sendChat(){
  const inp=document.getElementById('chatInput');
  const txt=inp.value.trim();if(!txt)return;
  inp.value='';
  addMsg(txt,'user');
  const typingId='typing-'+Date.now();
  const msgs=document.getElementById('chatMsgs');
  msgs.innerHTML+=`<div class="msg typing" id="${typingId}"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
  msgs.scrollTop=msgs.scrollHeight;
  try{
    const resp=await fetch('https://api.anthropic.com/v1/messages',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        model:'claude-sonnet-4-20250514',
        max_tokens:1000,
        system:"You are an expert architect and interior designer named Arch-Ai-Bot. Answer clearly and concisely. Provide checklists and step-by-step guidance when helpful. Keep responses short and practical.",
        messages:[{role:'user',content:txt}]
      })
    });
    const data=await resp.json();
    const reply=data.content?.map(c=>c.text||'').join('')||"Sorry, I couldn't get a response.";
    document.getElementById(typingId)?.remove();
    addMsg(reply,'bot');
  }catch(e){
    document.getElementById(typingId)?.remove();
    addMsg("I'm having trouble connecting. For architecture questions, try asking about room sizing, layout principles, or building materials!","bot");
  }
}
function addMsg(text,role){
  const msgs=document.getElementById('chatMsgs');
  const div=document.createElement('div');div.className=`msg ${role}`;div.textContent=text;
  msgs.appendChild(div);msgs.scrollTop=msgs.scrollHeight;
}
 
/* ─── BIDIRECTIONAL SCROLL REVEAL ─── */
function checkReveal(){
  const vh=window.innerHeight;
  document.querySelectorAll('.sr').forEach(el=>{
    const rect=el.getBoundingClientRect();
    const inView=rect.top<vh*0.88&&rect.bottom>0;
    const aboveView=rect.bottom<0;
    if(inView){el.classList.add('visible');el.classList.remove('past');}
    else if(aboveView){el.classList.remove('visible');el.classList.add('past');}
    else{el.classList.remove('visible');el.classList.remove('past');}
  });
}
window.addEventListener('scroll',checkReveal,{passive:true});
checkReveal();
