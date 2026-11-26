(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { input: null, baseline: null, shock: null, zero: null };
  const names = ["t2m", "tp", "u10", "v10", "msl", "tcwv"];
  const units = ["K", "mm/day", "m/s", "m/s", "Pa", "kg/m²"];
  const statusDot = $("status-dot");
  const statusText = $("status-text");
  const msg = $("message");
  const map = $("map");
  const ctx = map.getContext("2d");

  function setMessage(text, isError = false) {
    msg.textContent = text;
    msg.classList.toggle("error", isError);
  }
  function setReady(ready, text) {
    statusDot.classList.toggle("ready", ready);
    statusText.textContent = text;
  }
  function finiteTree(value, path) {
    path = path || "array";
    if (typeof value === "number") {
      if (!Number.isFinite(value)) throw new Error(path + " contains a non-finite number");
      return;
    }
    if (!Array.isArray(value)) throw new Error(path + " must be nested numeric lists");
    for (let i = 0; i < value.length; i++) finiteTree(value[i], path + "[" + i + "]");
  }
  function shapeOf(a) {
    const shape = []; let v = a;
    while (Array.isArray(v)) { shape.push(v.length); v = v[0]; }
    return shape;
  }
  function validateInput(data) {
    if (!data || typeof data !== "object") throw new Error("Input JSON must be an object");
    if (JSON.stringify(shapeOf(data.init_states)) !== JSON.stringify([3,6,32,36]))
      throw new Error("init_states must have shape [3,6,32,36]");
    if (!Array.isArray(data.doy_sincos) || data.doy_sincos.length !== 2 || !data.doy_sincos.every(Number.isFinite))
      throw new Error("doy_sincos must be [sin, cos] for the last input day");
    finiteTree(data.init_states, "init_states");
    if (data.perturbation !== undefined && data.perturbation !== null)
      throw new Error("Upload only init_states and doy_sincos; shocks are created by this UI");
    return { init_states: data.init_states, doy_sincos: data.doy_sincos };
  }
  async function checkHealth() {
    try {
      const res = await fetch("/health", { method: "POST" });
      const data = await res.json();
      if (!res.ok || data.status !== "ready") throw new Error((data.detail && data.detail.reason) || "Model is not ready");
      setReady(true, "Ready · " + data.model_name);
    } catch (err) {
      setReady(false, "Model weights not loaded");
      setMessage("API unavailable or model not ready: " + err.message + ". Add trained weights at weights/model.pt.", true);
    }
  }
  function makeShock() {
    const preset = $("shock-preset").value;
    const a = Array.from({length:6}, () => Array.from({length:32}, () => Array(36).fill(0)));
    const last = state.input.init_states[2];

    if (preset === "warm_ocean" || preset === "monsoon_stall") {
      const la0 = 8, la1 = 20;
      const lo0 = preset === "warm_ocean" ? 82 : 65;
      const lo1 = preset === "warm_ocean" ? 94 : 75;
      for (let y=0; y<32; y++) for (let x=0; x<36; x++) {
        const lat = 6.5 + y, lon = 65.5 + x;
        if (lat < la0 || lat > la1 || lon < lo0 || lon > lo1) continue;
        if (preset === "warm_ocean") {
          a[0][y][x] = 3.0;
          a[5][y][x] = 0.15 * last[5][y][x];
        } else {
          a[2][y][x] = -0.40 * last[2][y][x];
          a[3][y][x] = -0.40 * last[3][y][x];
        }
      }
      return a;
    }

    const channel = Number($("shock-channel").value);
    const delta = Number($("shock-delta").value);
    const la0 = Number($("lat-start").value), la1 = Number($("lat-end").value);
    const lo0 = Number($("lon-start").value), lo1 = Number($("lon-end").value);
    if (![channel, delta, la0, la1, lo0, lo1].every(Number.isFinite)) throw new Error("Shock controls must be finite numbers");
    if (la0 > la1 || lo0 > lo1) throw new Error("Shock start coordinates must not exceed end coordinates");
    for (let y=0; y<32; y++) for (let x=0; x<36; x++) {
      const lat=6.5+y, lon=65.5+x;
      if (lat>=la0 && lat<=la1 && lon>=lo0 && lon<=lo1) a[channel][y][x]=delta;
    }
    return a;
  }
  async function callSim(perturbation) {
    if (!state.input) throw new Error("Upload a real input window JSON file first");
    const steps=Number($("steps").value);
    if (!Number.isInteger(steps) || steps<1 || steps>14) throw new Error("Forecast steps must be 1–14");
    const payload=Object.assign({},state.input,{steps:steps,perturbation:perturbation});
    const res=await fetch("/simulate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const data=await res.json();
    if (!res.ok) throw new Error((data.detail && data.detail.reason) || data.detail || ("API HTTP "+res.status));
    if (!Array.isArray(data.rollout) || JSON.stringify(shapeOf(data.rollout))!==JSON.stringify([steps,6,32,36]))
      throw new Error("API returned an invalid rollout shape");
    finiteTree(data.rollout,"rollout");
    return data;
  }
  function val(rollout,day,ch,y,x) { return rollout[day][ch][y][x]; }
  function maxDiff(a,b,day,ch) {
    let m=0; for(let y=0;y<32;y++) for(let x=0;x<36;x++) m=Math.max(m,Math.abs(val(a,day,ch,y,x)-val(b,day,ch,y,x))); return m;
  }
  function color(v,lo,hi,diverging) {
    const t=Math.max(0,Math.min(1,(v-lo)/Math.max(hi-lo,1e-12)));
    if(diverging) {
      if(t<0.5) { const q=t*2; return [Math.round(35+215*q),Math.round(80+145*q),245]; }
      const q=(t-0.5)*2; return [250,Math.round(225-175*q),Math.round(245-210*q)];
    }
    return [Math.round(25+225*t),Math.round(75+115*(1-Math.abs(t-0.55))),Math.round(170+65*(1-t))];
  }
  function draw() {
    const day=Number($("day").value)-1, ch=Number($("channel").value), mode=$("view-mode").value;
    const source=mode==="shock"?state.shock:state.baseline;
    if(!source || (mode==="difference" && (!state.baseline || !state.shock))) {
      ctx.clearRect(0,0,map.width,map.height); ctx.fillStyle="#aab5c5"; ctx.font="18px system-ui";
      ctx.fillText("Run baseline and shocked forecasts to compare.",30,55); return;
    }
    if(day>=source.length) { $("day").value=source.length; return draw(); }
    const vals=[];
    for(let y=0;y<32;y++) for(let x=0;x<36;x++) vals.push(mode==="difference" ? val(state.shock,day,ch,y,x)-val(state.baseline,day,ch,y,x) : val(source,day,ch,y,x));
    let lo=Math.min.apply(null,vals), hi=Math.max.apply(null,vals);
    if(mode==="difference") { const m=Math.max(Math.abs(lo),Math.abs(hi),1e-9); lo=-m; hi=m; }
    else if(hi===lo) { lo-=1; hi+=1; }
    ctx.clearRect(0,0,map.width,map.height);
    const cw=map.width/36, chh=map.height/32;
    for(let y=0;y<32;y++) for(let x=0;x<36;x++) {
      const v=mode==="difference" ? val(state.shock,day,ch,y,x)-val(state.baseline,day,ch,y,x) : val(source,day,ch,y,x);
      const rgb=color(v,lo,hi,mode==="difference");
      ctx.fillStyle="rgb("+rgb.join(",")+")"; ctx.fillRect(x*cw,(31-y)*chh,Math.ceil(cw),Math.ceil(chh));
    }
    ctx.strokeStyle="rgba(255,255,255,.16)"; ctx.lineWidth=1;
    for(let x=0;x<=36;x+=6) { ctx.beginPath();ctx.moveTo(x*cw,0);ctx.lineTo(x*cw,map.height);ctx.stroke(); }
    for(let y=0;y<=32;y+=4) { ctx.beginPath();ctx.moveTo(0,y*chh);ctx.lineTo(map.width,y*chh);ctx.stroke(); }
    $("result-label").textContent=(mode==="difference"?"Difference":mode==="shock"?"Counterfactual":"Baseline")+" · Day "+(day+1);
    const maxD=state.baseline&&state.shock?maxDiff(state.baseline,state.shock,day,ch).toFixed(4):"—";
    $("metrics").innerHTML='<div class="metric"><span>Min</span><strong>'+lo.toFixed(3)+' '+(mode==="difference"?"":units[ch])+'</strong></div><div class="metric"><span>Max</span><strong>'+hi.toFixed(3)+' '+(mode==="difference"?"":units[ch])+'</strong></div><div class="metric"><span>Max |difference|</span><strong>'+maxD+'</strong></div>';
  }
  $("window-file").addEventListener("change",async e=>{
    const file=e.target.files && e.target.files[0]; if(!file)return;
    try {
      state.input=validateInput(JSON.parse(await file.text()));
      state.baseline=state.shock=state.zero=null;
      $("run-shock").disabled=$("run-zero").disabled=false;
      $("day").max=$("steps").value; setMessage("Loaded "+file.name+". Run baseline forecast.");draw();
    } catch(err) { state.input=null;setMessage(err.message,true); }
  });
  $("run-baseline").addEventListener("click",async()=>{
    try {
      $("run-baseline").disabled=true;setMessage("Running live baseline simulation…");
      const out=await callSim(null);state.baseline=out.rollout;
      $("baseline-time").textContent="Model inference: "+out.inference_ms.toFixed(2)+" ms";
      $("day").max=$("steps").value;$("day").value=1;$("view-mode").value="baseline";
      setMessage("Baseline forecast received from live FORKCAST model.");draw();
    } catch(err) {setMessage(err.message,true);} finally {$("run-baseline").disabled=false;}
  });
  $("run-shock").addEventListener("click",async()=>{
    try {
      if(!state.baseline)throw new Error("Run baseline forecast first");
      $("run-shock").disabled=true;setMessage("Running counterfactual simulation…");
      const out=await callSim(makeShock());state.shock=out.rollout;
      $("shock-time").textContent="Model inference: "+out.inference_ms.toFixed(2)+" ms";
      $("view-mode").value="shock";setMessage("Shocked forecast received from live FORKCAST model.");draw();
    } catch(err) {setMessage(err.message,true);} finally {$("run-shock").disabled=false;}
  });
  $("run-zero").addEventListener("click",async()=>{
    try {
      if(!state.baseline)throw new Error("Run baseline forecast first");
      $("run-zero").disabled=true;setMessage("Checking zero-shock equality…");
      const z=Array.from({length:6},()=>Array.from({length:32},()=>Array(36).fill(0)));
      const out=await callSim(z);state.zero=out.rollout;state.shock=state.zero;
      let global=0;for(let d=0;d<state.baseline.length;d++)for(let c=0;c<6;c++)global=Math.max(global,maxDiff(state.baseline,state.zero,d,c));
      setMessage("Zero-shock max absolute difference: "+global.toExponential(3)+(global<=1e-3?" · within raw tolerance; also verify channel-std contract":" · CHECK FAILED"),global>1e-3);
      $("view-mode").value="difference";draw();
    } catch(err) {setMessage(err.message,true);} finally {$("run-zero").disabled=false;}
  });
  ["channel","day","view-mode"].forEach(id=>$(id).addEventListener("change",draw));
  $("steps").addEventListener("change",()=>{$("day").max=$("steps").value;if(Number($("day").value)>Number($("steps").value))$("day").value=$("steps").value;});
  checkHealth();draw();
})();