import React,{useEffect,useMemo,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Activity,ShieldCheck,Home,BrainCircuit,UploadCloud,BarChart3,Info,ChevronRight,CheckCircle2,AlertTriangle,Clock3,Database,ArrowUpRight,Download,RefreshCw} from 'lucide-react';
import {ResponsiveContainer,LineChart,Line,XAxis,YAxis,Tooltip,CartesianGrid,ReferenceLine,BarChart,Bar,Legend} from 'recharts';
import './styles.css';

const API=import.meta.env.VITE_API_URL||'https://credit-risk-api-ckia.onrender.com';
const FEATURE_LABELS={loan_amnt:'Loan amount',term:'Term',int_rate:'Interest rate',grade:'Credit grade',emp_length:'Employment length',home_ownership:'Home ownership',annual_inc:'Annual income',purpose:'Loan purpose',dti:'Debt-to-income',delinq_2yrs:'Delinquencies (2y)',fico_score:'FICO score',inq_last_6mths:'Inquiries (6m)',open_acc:'Open accounts',revol_util:'Revolving utilisation',total_acc:'Total accounts',tot_cur_bal:'Current balance',total_rev_hi_lim:'Revolving credit limit',num_tl_op_past_12m:'New accounts (12m)',acc_now_delinq:'Current delinquencies',pub_rec_bankruptcies:'Bankruptcies'};
const NAV=[['overview','Overview',Home],['predict','Credit prediction',BrainCircuit],['batch','Batch prediction',Database],['monitor','Trustworthiness',ShieldCheck],['research','Research dashboard',BarChart3],['models','Model management',RefreshCw],['about','About',Info]];

function format(v,d=3){return typeof v==='number'?v.toFixed(d):'—'}
function pct(v){return typeof v==='number'?`${(v*100).toFixed(2)}%`:'—'}
async function useApi(path,opts={}){
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),180000);
  try{
    const r=await fetch(`${API}${path}`,{...opts,signal:controller.signal});
    const text=await r.text();
    let data=null;
    try{data=text?JSON.parse(text):null}catch{data=null}
    if(!r.ok){
      const detail=data?.detail;
      if(typeof detail==='string') throw new Error(detail);
      if(detail&&typeof detail==='object') throw new Error(detail.message||JSON.stringify(detail));
      throw new Error(`Request failed (HTTP ${r.status})`);
    }
    return data;
  }catch(e){
    if(e?.name==='AbortError') throw new Error('The server took longer than 3 minutes. The dataset may be large or retraining may still be running. Check the backend console.');
    throw e;
  }finally{clearTimeout(timer)}
}

function Metric({label,value,hint,tone='neutral'}){return <div className={`metric ${tone}`}><div className="metric-label">{label}<span className={`dot ${tone}`}/></div><div className="metric-value">{value}</div><div className="metric-hint">{hint}</div></div>}
function StatusPill({ok=true,children}){return <div className={`status-pill ${ok?'ok':'bad'}`}><Activity size={14}/>{children}</div>}
function Shell({page,setPage,health,children}){return <div className="app"><aside className="sidebar"><div className="brand"><div className="brand-mark"><ShieldCheck size={24}/></div><div><b>Trustworthy AI</b><span>Credit Risk Platform</span></div></div><nav>{NAV.map(([id,label,Icon])=><button key={id} onClick={()=>setPage(id)} className={page===id?'active':''}><Icon size={18}/><span>{label}</span><ChevronRight size={15}/></button>)}</nav><div className="side-foot"><span className="live-dot"/> <div><b>System online</b><small>{health?.active_model||'Loading model…'}</small></div></div></aside><main className="main"><header className="topbar"><div><span className="eyebrow">CREDIT RISK PLATFORM</span><h1>{NAV.find(x=>x[0]===page)?.[1]||'Dashboard'}</h1></div><div className="top-actions"><StatusPill ok={health?.status==='healthy'}>{health?.status==='healthy'?'Live':'Offline'}</StatusPill><span className="version-pill">Model {health?.active_model_version||'—'}</span></div></header>{children}</main></div>}

function Overview({go}){const [research,setResearch]=useState(null);useEffect(()=>{useApi('/api/research').then(setResearch).catch(()=>{});},[]);const c=research?.coverage?.lightgbm?.static||[];const chart=(research?.coverage?.years||[]).map((y,i)=>({year:y,static:c[i]??null,rolling:research?.coverage?.lightgbm?.rolling?.[i],aci:research?.coverage?.lightgbm?.aci?.[i]}));const t=research?.verified_trustworthiness?.after;return <div className="page"><section className="hero"><div><span className="eyebrow">TRUSTWORTHINESS-FIRST CREDIT SCORING</span><h2>Predict risk. Monitor trust. Respond to drift.</h2><p>One public interface for live borrower scoring and the verified evidence from the temporal credit-risk study.</p><button className="primary" onClick={()=>go('predict')}>Run a credit assessment <ArrowUpRight size={16}/></button></div><div className="hero-state"><ShieldCheck size={30}/><b>Model ready</b><span>20 deployed features</span><strong>VALID</strong></div></section><div className="metric-grid four"><Metric label="ROC-AUC" value={format(t?.auc,4)} hint="verified after retraining" tone="good"/><Metric label="ECE" value={format(t?.ece,4)} hint="threshold ≤ 0.08" tone="good"/><Metric label="CP coverage" value={pct(t?.coverage)} hint="90% nominal target" tone="good"/><Metric label="PSI" value={format(t?.psi,4)} hint="threshold ≤ 0.25" tone="good"/></div><div className="grid two"><section className="panel chart"><div className="panel-head"><div><span className="eyebrow">VERIFIED RESEARCH</span><h3>LightGBM coverage over time</h3></div><span className="tag">2017–2019</span></div><div className="chart-body"><ResponsiveContainer width="100%" height={270}><LineChart data={chart}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="year"/><YAxis domain={[0.82,0.92]} tickFormatter={x=>`${Math.round(x*100)}%`}/><Tooltip formatter={x=>`${(x*100).toFixed(2)}%`}/><ReferenceLine y={0.9} stroke="#94a3b8" strokeDasharray="5 5"/><Line type="monotone" dataKey="static" name="Static CP" stroke="#15233a" strokeWidth={3} dot={{r:4}}/><Line type="monotone" dataKey="rolling" name="Rolling CP" stroke="#4b7bec" strokeWidth={2}/><Line type="monotone" dataKey="aci" name="ACI" stroke="#27a57d" strokeWidth={2}/></LineChart></ResponsiveContainer></div></section><section className="panel"><div className="panel-head"><div><span className="eyebrow">MODEL MANAGEMENT</span><h3>Retraining outcome</h3></div><span className="tag good">Accepted</span></div><div className="comparison"><div><span>Before</span><strong>0.6891</strong><small>ROC-AUC</small></div><div className="arrow">→</div><div><span>After</span><strong>0.7385</strong><small>ROC-AUC</small></div></div><div className="list"><div><span>ECE</span><b>0.0380</b></div><div><span>Coverage</span><b>91.55%</b></div><div><span>PSI</span><b>0.0030</b></div><div><span>SHAP Drift</span><b>0.0261</b></div></div><button className="secondary wide" onClick={()=>go('monitor')}>Open trustworthiness <ChevronRight size={15}/></button></section></div></div>}

function Predict(){const fields=['loan_amnt','term','int_rate','grade','emp_length','home_ownership','annual_inc','purpose','dti','delinq_2yrs','fico_score','inq_last_6mths','open_acc','revol_util','total_acc','tot_cur_bal','total_rev_hi_lim','num_tl_op_past_12m','acc_now_delinq','pub_rec_bankruptcies'];const initial=Object.fromEntries(fields.map(f=>[f,'']));const [form,setForm]=useState(initial);const [result,setResult]=useState(null);const [busy,setBusy]=useState(false);const [err,setErr]=useState('');async function submit(){setBusy(true);setErr('');try{const features=Object.fromEntries(fields.map(f=>[f,Number(form[f]) ]));const r=await useApi('/api/predict',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({features})});setResult(r)}catch(e){setErr(e.message)}finally{setBusy(false)}}return <div className="page"><div className="section-head"><div><span className="eyebrow">LIVE INFERENCE</span><h2>Credit risk assessment</h2><p>Enter the 20 deployed model features. Prediction is generated by the production LightGBM model.</p></div><span className="tag">Live</span></div><div className="grid predict-grid"><section className="panel form-panel"><div className="form-section"><span>Loan & borrower information</span><div className="form-grid">{fields.map(f=><label key={f}>{FEATURE_LABELS[f]}<input value={form[f]} type="number" step="any" onChange={e=>setForm({...form,[f]:e.target.value})}/></label>)}</div></div>{err&&<div className="error"><AlertTriangle size={16}/>{err}</div>}<button className="primary wide" disabled={busy||fields.some(f=>form[f]==='')} onClick={submit}>{busy?'Scoring…':'Assess credit risk'} <ArrowUpRight size={16}/></button></section><section className="panel result-panel"><div className="panel-head"><div><span className="eyebrow">ASSESSMENT</span><h3>Prediction result</h3></div>{result&&<span className={`risk ${result.risk_category.toLowerCase().split(' ')[0]}`}>{result.risk_category}</span>}</div>{!result?<div className="empty"><BrainCircuit size={28}/><b>Ready for assessment</b><span>Complete all fields and submit the application.</span></div>:<div><div className="prob"><span>Default probability</span><strong>{pct(result.default_probability)}</strong><div className="bar"><i style={{width:`${result.default_probability*100}%`}}/></div></div><div className="interval"><span>Conformal prediction</span><strong>Not reported for the retrained production model</strong><small>{result.conformal_note}</small></div><div className="meta"><div><span>Model</span><b>{result.model}</b></div><div><span>Version</span><b>{result.model_version}</b></div></div></div>}</section></div></div>}

function Batch(){const [file,setFile]=useState(null),[data,setData]=useState(null),[busy,setBusy]=useState(false),[err,setErr]=useState('');async function run(){if(!file)return;setBusy(true);setErr('');try{const fd=new FormData();fd.append('file',file);setData(await useApi('/api/predict/batch',{method:'POST',body:fd}))}catch(e){setErr(e.message)}finally{setBusy(false)}}function download(){const a=document.createElement('a');a.href=`data:text/csv;base64,${data.csv_base64}`;a.download='credit-risk-predictions.csv';a.click()}return <div className="page"><div className="section-head"><div><span className="eyebrow">PORTFOLIO SCORING</span><h2>Batch prediction</h2><p>Upload a CSV containing the 20 deployed features.</p></div></div><div className="grid two"><section className="panel upload-panel"><label className="drop"><UploadCloud size={30}/><b>{file?file.name:'Choose a CSV file'}</b><span>Drop file here or browse</span><input type="file" accept=".csv" onChange={e=>setFile(e.target.files?.[0]||null)}/></label>{err&&<div className="error"><AlertTriangle size={16}/>{err}</div>}<button className="primary wide" disabled={!file||busy} onClick={run}>{busy?'Processing…':'Run batch prediction'}</button></section><section className="panel"><div className="panel-head"><div><span className="eyebrow">SUMMARY</span><h3>Portfolio risk mix</h3></div>{data&&<button className="icon-btn" onClick={download}><Download size={16}/></button>}</div>{!data?<div className="empty"><Database size={28}/><b>No portfolio loaded</b><span>Results and risk mix will appear here.</span></div>:<><div className="metric-grid three compact"><Metric label="Low risk" value={data.summary.low} hint="< 30%" tone="good"/><Metric label="Medium" value={data.summary.medium} hint="30–60%" tone="warn"/><Metric label="High risk" value={data.summary.high} hint="≥ 60%" tone="bad"/></div><div className="table-wrap"> <table><thead><tr>{Object.keys(data.preview?.[0]||{}).slice(-4).map(k=><th key={k}>{k}</th>)}</tr></thead><tbody>{(data.preview||[]).slice(0,8).map((r,i)=><tr key={i}>{Object.entries(r).slice(-4).map(([k,v])=><td key={k}>{typeof v==='number'?v.toFixed?.(4)??v:v}</td>)}</tr>)}</tbody></table></div></>}</section></div></div>}

function Monitor(){
  const verified={auc:0.7385,brier:0.1610,logloss:0.4895,ece:0.0380,coverage:0.9155,psi:0.0030,shap:0.0261};
  const [file,setFile]=useState(null),[live,setLive]=useState(null),[busy,setBusy]=useState(false),[err,setErr]=useState('');
  async function evaluate(){
    if(!file)return;
    setBusy(true);setErr('');setLive(null);
    try{
      const fd=new FormData();fd.append('file',file);fd.append('auto_retrain','true');
      const result=await useApi('/api/monitor',{method:'POST',body:fd});
      setLive(result);
      if(result?.retraining?.deployment_status==='DEPLOYED') window.dispatchEvent(new Event('model-deployed'));
    }catch(e){setErr(e.message)}finally{setBusy(false)}
  }
  const cards=[
    ['ROC-AUC',verified.auc.toFixed(4),'threshold ≥ 0.70','good'],
    ['Brier Score',verified.brier.toFixed(4),'supporting metric','good'],
    ['LogLoss',verified.logloss.toFixed(4),'supporting metric','good'],
    ['ECE',verified.ece.toFixed(4),'threshold ≤ 0.08','good'],
    ['CP Coverage',pct(verified.coverage),'threshold ≥ 90%','good'],
    ['PSI',verified.psi.toFixed(4),'threshold ≤ 0.25','good'],
    ['SHAP Drift',verified.shap.toFixed(4),'threshold ≤ 0.15','good']
  ];
  const m=live?.metrics||{},f=live?.threshold_flags||{};
  const liveCards=[
    ['ROC-AUC',m.roc_auc,'concern below 0.70',f.roc_auc],
    ['Brier Score',m.brier_score,'supporting metric',false],
    ['LogLoss',m.log_loss,'supporting metric',false],
    ['ECE',m.ece,'concern above 0.08',f.ece],
    ['CP Coverage',typeof m.cp_coverage==='number'?pct(m.cp_coverage):'—','concern below 90%',f.cp_coverage],
    ['PSI',m.psi,'concern above 0.25',f.psi],
    ['SHAP Drift',m.shap_drift,'concern above 0.15',f.shap_drift]
  ];
  const retr=live?.retraining;
  const after=live?.after_retraining;
  return <div className="page">
    <section className="panel monitor-hero"><div><span className="eyebrow">VERIFIED TRUSTWORTHINESS</span><h2>Model health</h2><p>Verified thesis results are shown above. The live check evaluates uploaded labelled data using the seven-indicator monitoring workflow.</p></div><div className="decision valid"><CheckCircle2 size={18}/><b>VALID</b></div></section>
    <div className="metric-grid four">{cards.map(([a,b,c,t])=><Metric key={a} label={a} value={b} hint={c} tone={t}/>)}</div>
    <div className="grid two">
      <section className="panel decision-box"><div className="decision-line"><CheckCircle2/><div><span>Verified retraining outcome</span><small>Current thesis result: ROC-AUC improved from 0.6891 to 0.7385.</small></div></div><div className="compare-row"><div><span>Before retraining</span><b>ROC-AUC 0.6891</b></div><div className="arrow">→</div><div><span>After retraining</span><b>ROC-AUC 0.7385</b></div></div></section>
      <section className="panel upload-panel"><div className="panel-head"><div><span className="eyebrow">LIVE CHECK</span><h3>Evaluate labelled data</h3></div></div><label className="compact-drop"><UploadCloud size={18}/><span>{file?file.name:'Choose monitoring CSV'}</span><input type="file" accept=".csv" onChange={e=>{setFile(e.target.files?.[0]||null);setErr('');setLive(null)}}/></label>{err&&<div className="error"><AlertTriangle size={16}/>{err}</div>}<button className="secondary wide" disabled={!file||busy} onClick={evaluate}>{busy?'Evaluating 7 indicators + retraining…':'Evaluate dataset'}</button><small className="helper">Automatic retraining is enabled when a threshold-based degradation signal is detected.</small></section>
    </div>
    {live&&<section className="panel live-result"><div className="panel-head"><div><span className="eyebrow">LIVE RESULT</span><h3>{live.decision||'RESULT'}</h3></div><span className={`tag ${live.decision==='RETRAIN'?'bad':'good'}`}>7/7 indicators</span></div>
      <div className="metric-grid four compact">{liveCards.map(([a,b,c,bad])=><Metric key={a} label={a} value={typeof b==='number'?format(b,6):b??'—'} hint={bad?c:'passes / supporting'} tone={bad?'bad':'good'}/>)}</div>
      <div className="notice"><Clock3 size={16}/><span>Model: {live.model||'—'} · Rows evaluated: {live.rows?.toLocaleString?.()||live.rows||'—'} · Threshold-based: 5 · Supporting: 2.</span></div>
      {live.decision==='RETRAIN'&&<div className={`notice ${retr?.deployment_status==='DEPLOYED'?'success-notice':''}`}><RefreshCw size={16}/><span><b>Retraining triggered.</b> {retr?.deployment_status||'Processing'}{retr?.candidate_holdout_roc_auc!=null&&<> · Current holdout ROC-AUC: {format(retr.current_holdout_roc_auc,4)} · Candidate: {format(retr.candidate_holdout_roc_auc,4)}</>}</span></div>}
      {retr?.status==='FAILED'&&<div className="error"><AlertTriangle size={16}/><span>Monitoring completed, but automatic retraining failed: {typeof retr.error==='string'?retr.error:JSON.stringify(retr.error)}</span></div>}
      {retr&&retr.current_holdout_roc_auc!=null&&retr.candidate_holdout_roc_auc!=null&&<section className="panel decision-box nested"><div className="panel-head"><div><span className="eyebrow">MODEL ACCEPTANCE</span><h3>Current vs candidate</h3></div><span className={`tag ${retr.accepted?'good':'bad'}`}>{retr.accepted?'ACCEPTED':'REJECTED'}</span></div><div className="compare-row"><div><span>Current production model</span><b>ROC-AUC {format(retr.current_holdout_roc_auc,4)}</b><small>{retr.current_model}</small></div><div className="arrow">→</div><div><span>Candidate model</span><b>ROC-AUC {format(retr.candidate_holdout_roc_auc,4)}</b><small>{retr.candidate_model||'Not deployed'}</small></div></div><div className={`notice ${retr.accepted?'success-notice':''}`}><CheckCircle2 size={16}/><span><b>{retr.deployment_status==='DEPLOYED'?'Candidate accepted and deployed.':'Current model retained.'}</b> {retr.roc_auc_change!=null&&<>ROC-AUC change: {retr.roc_auc_change>=0?'+':''}{format(retr.roc_auc_change,4)}.</>}</span></div></section>}
      {after?.metrics&&<div className="notice success-notice"><CheckCircle2 size={16}/><span><b>Production model updated:</b> {after.model} · ROC-AUC {format(after.metrics.roc_auc,4)} · ECE {format(after.metrics.ece,4)} · CP Coverage {pct(after.metrics.cp_coverage)}.</span></div>}
      {after?.metrics&&<div className="metric-grid four compact"><Metric label="After ROC-AUC" value={format(after.metrics.roc_auc,4)} hint="same uploaded monitoring data" tone="good"/><Metric label="After Brier" value={format(after.metrics.brier_score,4)} hint="lower is better" tone="good"/><Metric label="After LogLoss" value={format(after.metrics.log_loss,4)} hint="lower is better" tone="good"/><Metric label="After ECE" value={format(after.metrics.ece,4)} hint="threshold ≤ 0.08" tone="good"/></div>}
      <div className="notice"><Info size={16}/><span>{live.conformal_recalibration?.note||'Conformal monitoring uses the available thesis q-hat; no new post-retraining q-hat is claimed.'}</span></div>
    </section>}
  </div>
}

function ModelManagement(){
  const [data,setData]=useState(null),[busy,setBusy]=useState(false),[err,setErr]=useState('');
  const [confirm,setConfirm]=useState(null);
  async function load(){
    try{setData(await useApi('/api/models'));setErr('')}catch(e){setErr(e.message)}
  }
  useEffect(()=>{load();const refresh=()=>load();window.addEventListener('model-deployed',refresh);return()=>window.removeEventListener('model-deployed',refresh)},[]);
  async function rollback(filename){
    setBusy(true);setErr('');
    try{
      await useApi('/api/models/rollback',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model:filename})});
      setConfirm(null);
      await load();
      window.dispatchEvent(new Event('model-deployed'));
    }catch(e){setErr(e.message)}finally{setBusy(false)}
  }
  const models=data?.models||[];
  return <div className="page">
    <div className="section-head"><div><span className="eyebrow">MODEL MANAGEMENT</span><h2>Model versions</h2><p>Every accepted production model is preserved. The active model can be rolled back without deleting previous versions.</p></div><button className="secondary" onClick={load} disabled={busy}><RefreshCw size={16}/> Refresh</button></div>
    {err&&<div className="error"><AlertTriangle size={16}/>{err}</div>}
    <div className="grid two">
      <section className="panel">
        <div className="panel-head"><div><span className="eyebrow">ACTIVE MODEL</span><h3>{data?.active_model||'Loading…'}</h3></div><span className="tag good">ACTIVE</span></div>
        <div className="list"><div><span>Version</span><b>{data?.active_model?data.active_model.replace(/^production_model_/,'').replace(/\.joblib$/,''):'—'}</b></div><div><span>Previous model</span><b>{data?.previous_model||'—'}</b></div><div><span>Baseline</span><b>{data?.baseline_model||'—'}</b></div></div>
      </section>
      <section className="panel">
        <div className="panel-head"><div><span className="eyebrow">ROLLBACK</span><h3>Restore a previous version</h3></div></div>
        <p className="helper">Rollback changes only the active model pointer. Existing model files remain preserved.</p>
        <select className="rollback-select" value={confirm||''} onChange={e=>setConfirm(e.target.value||null)}>
          <option value="">Select a model…</option>
          {models.filter(x=>x.filename!==data?.active_model).map(x=><option key={x.filename} value={x.filename}>{x.filename}{x.filename===data?.baseline_model?' (baseline)':''}</option>)}
        </select>
        <button className="secondary wide" disabled={!confirm||busy} onClick={()=>rollback(confirm)}>{busy?'Rolling back…':'Confirm rollback'}</button>
      </section>
    </div>
    <section className="panel"><div className="panel-head"><div><span className="eyebrow">VERSION HISTORY</span><h3>Preserved model files</h3></div><span className="tag">{models.length} version{models.length===1?'':'s'}</span></div>
      <div className="table-wrap"><table><thead><tr><th>Model</th><th>Type</th><th>ROC-AUC</th><th>Status</th></tr></thead><tbody>{models.map((m,i)=><tr key={m.filename||i}><td>{m.filename}</td><td>{m.type||'production'}</td><td>{m.roc_auc!=null?format(m.roc_auc,4):'—'}</td><td><span className={`tag ${m.active?'good':''}`}>{m.active?'ACTIVE':'ARCHIVED'}</span></td></tr>)}</tbody></table></div>
    </section>
    {confirm&&<div className="notice"><AlertTriangle size={16}/><span>Selected rollback target: <b>{confirm}</b>. Click “Confirm rollback” to activate it.</span></div>}
  </div>
}

function Research(){const [r,setR]=useState(null);useEffect(()=>{useApi('/api/research').then(setR).catch(()=>{});},[]);if(!r)return <div className="page"><div className="panel empty"><RefreshCw/><b>Loading verified research</b></div></div>;const cov=(r.coverage.years||[]).map((y,i)=>({year:y,static:r.coverage.lightgbm.static[i],rolling:r.coverage.lightgbm.rolling[i],aci:r.coverage.lightgbm.aci[i]}));const psi=r.psi.synthetic.years.map((y,i)=>({year:y,psi:r.psi.synthetic.average[i]}));const shap=r.shap.drift_scores.slice(0,10).map(([feature,score])=>({feature,score}));const stress=r.synthetic_stress.years.map((y,i)=>({year:y,Static:r.synthetic_stress.static[i],Rolling:r.synthetic_stress.rolling[i],ACI:r.synthetic_stress.aci[i]}));return <div className="page"><div className="section-head"><div><span className="eyebrow">VERIFIED EVIDENCE</span><h2>Research dashboard</h2><p>Interactive views of the final experimental results.</p></div></div><div className="grid two"><section className="panel chart"><div className="panel-head"><div><span className="eyebrow">COVERAGE</span><h3>Static vs Rolling vs ACI</h3></div><span className="tag">90% target</span></div><div className="chart-body"><ResponsiveContainer width="100%" height={290}><LineChart data={cov}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="year"/><YAxis domain={[0.82,0.92]} tickFormatter={x=>`${Math.round(x*100)}%`}/><Tooltip formatter={x=>`${(x*100).toFixed(2)}%`}/><ReferenceLine y={0.9} stroke="#94a3b8" strokeDasharray="5 5"/><Line dataKey="static" name="Static CP" stroke="#15233a" strokeWidth={3}/><Line dataKey="rolling" name="Rolling CP" stroke="#4b7bec" strokeWidth={2}/><Line dataKey="aci" name="ACI" stroke="#27a57d" strokeWidth={2}/></LineChart></ResponsiveContainer></div></section><section className="panel chart"><div className="panel-head"><div><span className="eyebrow">SYNTHETIC STRESS</span><h3>Average PSI, 2020–2026</h3></div><span className="tag">Stress test</span></div><div className="chart-body"><ResponsiveContainer width="100%" height={290}><LineChart data={psi}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="year"/><YAxis/><Tooltip/><ReferenceLine y={0.25} stroke="#94a3b8" strokeDasharray="5 5"/><Line dataKey="psi" name="Average PSI" stroke="#8c4bbf" strokeWidth={3}/></LineChart></ResponsiveContainer></div></section></div><div className="grid two"><section className="panel chart"><div className="panel-head"><div><span className="eyebrow">SHAP DRIFT</span><h3>Highest explanation instability</h3></div><span className="tag">Threshold 0.15</span></div><div className="chart-body"><ResponsiveContainer width="100%" height={310}><BarChart data={shap} layout="vertical" margin={{left:30,right:20}}><CartesianGrid strokeDasharray="3 3" horizontal={false}/><XAxis type="number"/><YAxis type="category" dataKey="feature" width={105}/><Tooltip/><ReferenceLine x={0.15} stroke="#94a3b8" strokeDasharray="5 5"/><Bar dataKey="score" name="SHAP Drift" fill="#2f6fdb"/></BarChart></ResponsiveContainer></div></section><section className="panel chart"><div className="panel-head"><div><span className="eyebrow">STRESS COVERAGE</span><h3>Static, Rolling and ACI</h3></div><span className="tag">2020–2026</span></div><div className="chart-body"><ResponsiveContainer width="100%" height={310}><LineChart data={stress}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="year"/><YAxis domain={[0.84,0.92]} tickFormatter={x=>`${Math.round(x*100)}%`}/><Tooltip formatter={x=>`${(x*100).toFixed(2)}%`}/><ReferenceLine y={0.85} stroke="#94a3b8" strokeDasharray="5 5"/><Line dataKey="Static" stroke="#15233a"/><Line dataKey="Rolling" stroke="#4b7bec"/><Line dataKey="ACI" stroke="#27a57d"/></LineChart></ResponsiveContainer></div></section></div><section className="panel evidence"><div className="panel-head"><div><span className="eyebrow">KEY FINDINGS</span><h3>Evidence at a glance</h3></div></div><div className="finding-grid"><div><span>Severe population drift</span><b>revol_util · PSI 0.3750</b></div><div><span>Highest explanation drift</span><b>acc_now_delinq · 1.4947</b></div><div><span>PSI–SHAP relationship</span><b>Not significant · p = 0.9248</b></div><div><span>Coverage recovery</span><b>ACI ≈ 90% across 2017–2019</b></div></div></section></div>}

function About(){return <div className="page"><section className="panel about"><span className="eyebrow">SYSTEM</span><h2>Trustworthy AI Credit Risk</h2><p>This public interface connects production LightGBM scoring with the empirical evidence described in the thesis: temporal conformal coverage, population drift, explanation drift, subgroup analysis, and drift-triggered retraining.</p><div className="about-grid"><div><b>Model</b><span>LightGBM</span></div><div><b>Features</b><span>20</span></div><div><b>Nominal coverage</b><span>90%</span></div><div><b>Decision thresholds</b><span>AUC · ECE · Coverage · PSI · SHAP</span></div></div></section></div>}

function App(){const [page,setPage]=useState('overview'),[health,setHealth]=useState(null);useEffect(()=>{const refresh=()=>useApi('/api/health').then(setHealth).catch(()=>{});refresh();window.addEventListener('model-deployed',refresh);return()=>window.removeEventListener('model-deployed',refresh);},[]);let body=page==='overview'?<Overview go={setPage}/>:page==='predict'?<Predict/>:page==='batch'?<Batch/>:page==='monitor'?<Monitor/>:page==='research'?<Research/>:page==='models'?<ModelManagement/>:<About/>;return <Shell page={page} setPage={setPage} health={health}>{body}</Shell>}
createRoot(document.getElementById('root')).render(<App/>);
