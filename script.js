/* ═══════════════════════════════════════════════════════════════
   VibeCheck — Application Logic
   ═══════════════════════════════════════════════════════════════ */

'use strict';

/* ─── STATE ─────────────────────────────────────────────────── */
const state = {
  videoDetails: null,
  results: [],
  maxComments: 50,
  heroMax: 50,
  tryMax: 50,
  activeTab: 'Positive',
  charts: { donut: null, timeline: null }
};

/* ─── YOUTUBE HELPERS ───────────────────────────────────────── */
function extractVideoId(url) {
  if (!url) return null;
  const patterns = [
    /(?:v=|\/embed\/|\/shorts\/|youtu\.be\/)([A-Za-z0-9_-]{11})/,
    /^([A-Za-z0-9_-]{11})$/
  ];
  for (const p of patterns) {
    const m = url.match(p);
    if (m) return m[1];
  }
  return null;
}

async function fetchYouTubeData(videoId) {
  const key = CONFIG.YOUTUBE_API_KEY;
  const [vidRes, cmtRes] = await Promise.all([
    fetch(`https://www.googleapis.com/youtube/v3/videos?id=${videoId}&part=snippet,statistics,contentDetails&key=${key}`),
    fetch(`https://www.googleapis.com/youtube/v3/commentThreads?videoId=${videoId}&part=snippet&maxResults=${state.maxComments}&order=relevance&key=${key}`)
  ]);

  if (!vidRes.ok) throw new Error('YouTube API error — check API key or video availability');
  if (!cmtRes.ok) {
    const errData = await cmtRes.json().catch(() => ({}));
    const msg = errData?.error?.message || 'Comments unavailable';
    throw new Error(msg);
  }

  const vidData = await vidRes.json();
  const cmtData = await cmtRes.json();

  if (!vidData.items?.length) throw new Error('Video not found or is private');

  const v = vidData.items[0];
  const duration = parseDuration(v.contentDetails?.duration || 'PT0S');

  return {
    videoDetails: {
      id: videoId,
      title:       v.snippet.title,
      channel:     v.snippet.channelTitle,
      published:   v.snippet.publishedAt,
      thumbnail:   v.snippet.thumbnails?.maxres?.url || v.snippet.thumbnails?.high?.url || v.snippet.thumbnails?.default?.url,
      views:       Number(v.statistics?.viewCount || 0),
      likes:       Number(v.statistics?.likeCount || 0),
      commentCount:Number(v.statistics?.commentCount || 0),
      duration
    },
    comments: (cmtData.items || []).map(item => {
      const c = item.snippet.topLevelComment.snippet;
      return { text: c.textDisplay || c.textOriginal || '', author: c.authorDisplayName, likes: c.likeCount, publishedAt: c.publishedAt };
    }).filter(c => c.text.trim().length > 0)
  };
}

function parseDuration(iso) {
  const m = iso.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/);
  if (!m) return '';
  const h = parseInt(m[1] || 0), min = parseInt(m[2] || 0), s = parseInt(m[3] || 0);
  if (h > 0) return `${h}:${String(min).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  return `${min}:${String(s).padStart(2,'0')}`;
}

/* ─── ML BACKEND ────────────────────────────────────────────── */
const BACKEND_URL = window.location.port === '8080' ? 'http://localhost:8000' : '';

async function runMLPrediction(comments) {
  const res = await fetch(`${BACKEND_URL}/predict/batch`, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ texts: comments.map(c => c.text) })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Backend error ${res.status}`);
  }
  return res.json();
}

/* ─── SAMPLE SIZE ───────────────────────────────────────────── */
function setSample(val, prefix) {
  state[`${prefix}Max`] = val;
  state.maxComments = val;
  document.querySelectorAll(`#${prefix}-sample-chips .chip`).forEach(c => {
    c.classList.toggle('chip--active', Number(c.dataset.val) === val);
  });
}

/* ─── ANALYSIS FLOW ─────────────────────────────────────────── */
async function startAnalysis(inputId) {
  const url = document.getElementById(inputId)?.value?.trim();
  if (!url) { showToast('Paste a YouTube URL first', 'error'); return; }

  const videoId = extractVideoId(url);
  if (!videoId) { showToast('Invalid YouTube URL — check the link', 'error'); return; }

  showLoading();

  try {
    // Step 1: backend connection check
    setStep('connect', 'active');
    try {
      const ping = await fetch(`${BACKEND_URL}/`);
      if (!ping.ok) throw new Error();
    } catch (e) {
      setStep('connect', 'fail');
      throw new Error(`Cannot connect to backend server. Please make sure the FastAPI server is running.`);
    }
    setStep('connect', 'done');


    // Step 2: fetch YouTube data
    setStep('fetch', 'active');
    let ytData;
    try {
      ytData = await fetchYouTubeData(videoId);
    } catch (e) {
      setStep('fetch', 'fail');
      throw new Error(`YouTube: ${e.message}`);
    }
    setStep('fetch', 'done');

    if (ytData.comments.length === 0) throw new Error('No comments found for this video');

    // Step 3: clean
    setStep('clean', 'active');
    await sleep(600);
    setStep('clean', 'done');

    // Step 4: run ML
    setStep('model', 'active');
    let mlResult;
    try {
      mlResult = await runMLPrediction(ytData.comments);
    } catch (e) {
      setStep('model', 'fail');
      throw new Error(`ML Model: ${e.message} — make sure the FastAPI server is running on port 8000`);
    }
    setStep('model', 'done');

    // Step 5: insights
    setStep('insights', 'active');
    const processed = processResults(ytData.comments, mlResult);
    state.videoDetails = ytData.videoDetails;
    state.results = processed;
    await sleep(500);
    setStep('insights', 'done');

    // Step 6: dashboard
    setStep('dashboard', 'active');
    await sleep(300);
    setStep('dashboard', 'done');
    await sleep(300);

    showDashboard();

  } catch (e) {
    showError('Analysis Failed', e.message);
  }
}

function processResults(comments, mlResult) {
  const preds = mlResult.predictions || mlResult || [];
  return comments.map((c, i) => ({
    ...c,
    sentiment: normalizeSentiment(preds[i]?.sentiment || preds[i]?.label || preds[i] || 'Neutral'),
    score: preds[i]?.score ?? preds[i]?.confidence ?? null
  }));
}

function normalizeSentiment(s) {
  if (!s) return 'Neutral';
  const v = String(s).toLowerCase();
  if (v.includes('pos')) return 'Positive';
  if (v.includes('neg')) return 'Negative';
  return 'Neutral';
}

/* ─── LOADING STEPS ─────────────────────────────────────────── */
const STEP_IDS = { connect:'lstep-connect', fetch:'lstep-fetch', clean:'lstep-clean', model:'lstep-model', insights:'lstep-insights', dashboard:'lstep-dashboard' };

function setStep(key, cls) {
  const el = document.getElementById(STEP_IDS[key]);
  if (!el) return;
  el.className = `lstep ${cls}`;
  const status = el.querySelector('.lstep__status');
  if (status) status.textContent = cls === 'done' ? '✓' : cls === 'fail' ? '✗' : '';
}

function resetSteps() {
  Object.values(STEP_IDS).forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.className = 'lstep'; const s = el.querySelector('.lstep__status'); if (s) s.textContent = '' }
  });
}

/* ─── PAGE TRANSITIONS ──────────────────────────────────────── */
function showLoading() {
  const landing = document.getElementById('landing-page');
  const loading = document.getElementById('loading-screen');
  const dash    = document.getElementById('dashboard');
  const error   = document.getElementById('error-screen');

  resetSteps();
  landing.style.display = 'none';
  error.style.display   = 'none';
  dash.style.display    = 'none';
  loading.style.display = 'flex';
  window.scrollTo(0,0);
}

function showDashboard() {
  document.getElementById('loading-screen').style.display = 'none';
  const dash = document.getElementById('dashboard');
  dash.style.display = 'block';
  window.scrollTo(0,0);
  renderDashboard();
}

function showError(title, msg) {
  document.getElementById('loading-screen').style.display = 'none';
  document.getElementById('error-screen').style.display   = 'flex';
  document.getElementById('err-title').textContent = title;
  document.getElementById('err-msg').textContent   = msg;
  document.getElementById('dashboard').style.display = 'none';
}

function goToLanding() {
  document.getElementById('dashboard').style.display     = 'none';
  document.getElementById('loading-screen').style.display = 'none';
  document.getElementById('error-screen').style.display   = 'none';
  const landing = document.getElementById('landing-page');
  landing.style.display = 'block';
  window.scrollTo(0,0);
  // Clear inputs
  ['hero-url','try-url'].forEach(id => { const el = document.getElementById(id); if (el) el.value = '' });
}

function scrollToTry() {
  document.getElementById('section-try')?.scrollIntoView({ behavior:'smooth' });
}
function scrollToTop(e) {
  e.preventDefault();
  window.scrollTo({ top:0, behavior:'smooth' });
}

/* ─── DASHBOARD RENDERING ───────────────────────────────────── */
function renderDashboard() {
  renderVideoCard();
  renderMetrics();
  renderCharts();
  renderBars();
  renderKeywords();
  renderEmotions();
  setCommentTab('Positive', document.querySelector('.etab--pos'));
  renderInsights();
  renderAISummary();
}

/* Video card */
function renderVideoCard() {
  const v = state.videoDetails;
  if (!v) return;
  document.getElementById('d-thumb').src = v.thumbnail || '';
  document.getElementById('d-title').textContent = v.title || 'Unknown Title';
  document.getElementById('d-channel').textContent = v.channel || '';
  document.getElementById('d-views').textContent = formatNum(v.views);
  document.getElementById('d-date').textContent = v.published ? new Date(v.published).toLocaleDateString('en-US',{year:'numeric',month:'short',day:'numeric'}) : '—';
  document.getElementById('d-comment-count').textContent = formatNum(v.commentCount);
  document.getElementById('d-analyzed').textContent = state.results.length + ' comments';
  const dur = document.getElementById('d-duration');
  if (dur) dur.textContent = v.duration || '';
}

/* Metrics */
function renderMetrics() {
  const total = state.results.length;
  const pos = state.results.filter(r=>r.sentiment==='Positive').length;
  const neu = state.results.filter(r=>r.sentiment==='Neutral').length;
  const neg = state.results.filter(r=>r.sentiment==='Negative').length;

  countUp('m-total',   total);
  countUp('m-positive',pos);
  countUp('m-neutral', neu);
  countUp('m-negative',neg);

  const pPct = total ? Math.round(pos/total*100) : 0;
  const neuPct = total ? Math.round(neu/total*100) : 0;
  const negPct = total ? Math.round(neg/total*100) : 0;

  setText('m-positive-pct', pPct + '%');
  setText('m-neutral-pct',  neuPct + '%');
  setText('m-negative-pct', negPct + '%');
}

/* Charts */
function renderCharts() {
  const total = state.results.length;
  const pos = state.results.filter(r=>r.sentiment==='Positive').length;
  const neu = state.results.filter(r=>r.sentiment==='Neutral').length;
  const neg = state.results.filter(r=>r.sentiment==='Negative').length;
  const pPct = total ? Math.round(pos/total*100) : 0;
  const neuPct = total ? Math.round(neu/total*100) : 0;
  const negPct = total ? Math.round(neg/total*100) : 0;

  // Donut legend
  setText('leg-pos', pPct+'%');
  setText('leg-neu', neuPct+'%');
  setText('leg-neg', negPct+'%');

  // Donut center label
  const dom = pos > neg && pos > neu ? '😊 Positive' : neg > pos && neg > neu ? '😡 Negative' : '😐 Neutral';
  setText('donut-label', dom.split(' ')[0]);

  // Donut chart
  if (state.charts.donut) state.charts.donut.destroy();
  const dCtx = document.getElementById('donut-chart');
  if (dCtx) {
    state.charts.donut = new Chart(dCtx, {
      type:'doughnut',
      data:{
        labels:['Positive','Neutral','Negative'],
        datasets:[{
          data:[pos, neu, neg],
          backgroundColor:['rgba(124,255,176,.8)','rgba(255,211,92,.8)','rgba(255,93,122,.8)'],
          borderColor:['#7CFFB0','#FFD35C','#FF5D7A'],
          borderWidth:2,
          hoverOffset:8
        }]
      },
      options:{
        cutout:'72%', responsive:true, maintainAspectRatio:true,
        plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>`${ctx.label}: ${ctx.parsed} (${Math.round(ctx.parsed/total*100)}%)`}}},
        animation:{duration:1200,easing:'easeInOutQuart'}
      }
    });
  }

  // Timeline chart
  if (state.charts.timeline) state.charts.timeline.destroy();
  const tCtx = document.getElementById('timeline-chart');
  if (tCtx) {
    const chunks = 10;
    const chunkSize = Math.max(1, Math.ceil(state.results.length / chunks));
    const labels = [], posData=[], neuData=[], negData=[];
    for (let i=0; i<state.results.length; i+=chunkSize) {
      const slice = state.results.slice(i, i+chunkSize);
      labels.push(`${i+1}–${Math.min(i+chunkSize,state.results.length)}`);
      posData.push(slice.filter(r=>r.sentiment==='Positive').length);
      neuData.push(slice.filter(r=>r.sentiment==='Neutral').length);
      negData.push(slice.filter(r=>r.sentiment==='Negative').length);
    }
    state.charts.timeline = new Chart(tCtx, {
      type:'line',
      data:{
        labels,
        datasets:[
          {label:'Positive',data:posData,borderColor:'#7CFFB0',backgroundColor:'rgba(124,255,176,.08)',tension:.4,fill:true,pointRadius:4,pointHoverRadius:6},
          {label:'Neutral', data:neuData,borderColor:'#FFD35C',backgroundColor:'rgba(255,211,92,.06)',tension:.4,fill:true,pointRadius:4,pointHoverRadius:6},
          {label:'Negative',data:negData,borderColor:'#FF5D7A',backgroundColor:'rgba(255,93,122,.08)',tension:.4,fill:true,pointRadius:4,pointHoverRadius:6}
        ]
      },
      options:{
        responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:true,labels:{color:'#B9A9D9',font:{family:'Space Mono',size:11}}}},
        scales:{
          x:{grid:{color:'rgba(245,239,255,.04)'},ticks:{color:'#7A6D9A',font:{size:10,family:'Space Mono'}}},
          y:{grid:{color:'rgba(245,239,255,.04)'},ticks:{color:'#7A6D9A',font:{size:10,family:'Space Mono'}}}
        },
        animation:{duration:1000,easing:'easeInOutQuart'}
      }
    });
  }
}

/* Bars */
function renderBars() {
  const total = state.results.length;
  const pos = state.results.filter(r=>r.sentiment==='Positive').length;
  const neu = state.results.filter(r=>r.sentiment==='Neutral').length;
  const neg = state.results.filter(r=>r.sentiment==='Negative').length;
  const pPct = total ? Math.round(pos/total*100) : 0;
  const neuPct = total ? Math.round(neu/total*100) : 0;
  const negPct = total ? Math.round(neg/total*100) : 0;

  setTimeout(() => {
    setBar('bar-pos', pPct); setText('bar-pos-pct', pPct+'%');
    setBar('bar-neu', neuPct); setText('bar-neu-pct', neuPct+'%');
    setBar('bar-neg', negPct); setText('bar-neg-pct', negPct+'%');
  }, 300);
}

function setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = pct+'%';
}

/* Keywords */
function renderKeywords() {
  const cloud = document.getElementById('keyword-cloud');
  if (!cloud) return;
  const stopwords = new Set(['the','a','an','is','it','was','in','on','of','to','and','or','but','not','this','that','for','are','be','at','as','by','with','from','have','has','had','do','does','did','will','would','could','should','may','might','can','its','their','they','them','we','us','you','your','he','she','his','her','him','our','my','me','i','so','if','up','out','more','than','about','into','no','all','also','just','been','like','get','got','what','who','when','how','very','too','much','well','even','after','because','which','there','then','some','any','one','two','three','make','know','think','see','go','going','really','already','actually','now']);
  const freq = {};
  state.results.forEach(r => {
    const words = r.text.replace(/<[^>]*>/g,'').toLowerCase().split(/\W+/);
    words.forEach(w => { if (w.length > 3 && !stopwords.has(w)) freq[w] = (freq[w]||0)+1 });
  });
  const sorted = Object.entries(freq).sort((a,b)=>b[1]-a[1]).slice(0,32);
  const max = sorted[0]?.[1] || 1;
  cloud.innerHTML = sorted.map(([w,c]) => {
    const size = 9 + Math.round((c/max)*8);
    return `<button class="kw-chip" style="font-size:${size}px;opacity:${0.5+0.5*(c/max)}">${w} <span style="color:var(--text3)">${c}</span></button>`;
  }).join('');
}

/* Emotions */
function renderEmotions() {
  const total = state.results.length || 1;
  const pos = state.results.filter(r=>r.sentiment==='Positive').length / total;
  const neg = state.results.filter(r=>r.sentiment==='Negative').length / total;
  const neu = state.results.filter(r=>r.sentiment==='Neutral').length / total;

  const happy        = Math.max(0, pos - 0.05);
  const neutral      = neu;
  const angry        = Math.max(0, neg * 0.6);
  const appreciation = Math.max(0, pos * 0.4);
  const curiosity    = Math.max(0, neu * 0.35);

  const normalize = arr => { const s = arr.reduce((a,b)=>a+b,0); return s > 0 ? arr.map(v=>v/s) : arr };
  const [nHappy, nNeutral, nAngry, nAppreciation, nCuriosity] = normalize([happy, neutral, angry, appreciation, curiosity]);

  setText('emo-happy',        Math.round(nHappy*100)+'%');
  setText('emo-neutral',      Math.round(nNeutral*100)+'%');
  setText('emo-angry',        Math.round(nAngry*100)+'%');
  setText('emo-appreciation', Math.round(nAppreciation*100)+'%');
  setText('emo-curiosity',    Math.round(nCuriosity*100)+'%');
}

/* Comments tab */
function setCommentTab(tab, btn) {
  state.activeTab = tab;
  document.querySelectorAll('.etab').forEach(b => b.classList.remove('etab--active'));
  if (btn) btn.classList.add('etab--active');
  renderComments();
}

function renderComments() {
  const list = document.getElementById('comment-list');
  if (!list) return;
  const filtered = state.results.filter(r => r.sentiment === state.activeTab);
  if (!filtered.length) {
    list.innerHTML = `<div style="color:var(--text3);font-family:var(--font-mono);font-size:12px;text-align:center;padding:40px 0;text-transform:uppercase;letter-spacing:.1em">No ${state.activeTab} comments found</div>`;
    return;
  }

  const color = state.activeTab === 'Positive' ? 'var(--positive)' : state.activeTab === 'Negative' ? 'var(--negative)' : 'var(--neutral)';
  list.innerHTML = filtered.slice(0,50).map(c => {
    const score = c.score != null ? `${Math.round(c.score*100)}% conf` : '';
    const date = c.publishedAt ? new Date(c.publishedAt).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'}) : '';
    const text = c.text.replace(/<[^>]*>/g,'').slice(0,400);
    return `<div class="comment-item">
      <div class="comment-item__top">
        <span class="comment-author">${escHtml(c.author||'Anonymous')}</span>
        <div class="comment-meta">
          ${score ? `<span class="confidence-badge">${score}</span>` : ''}
          <span class="sentiment-tag" style="color:${color}">${c.sentiment}</span>
        </div>
      </div>
      <p class="comment-text">${escHtml(text)}</p>
      <div class="comment-footer">
        ${c.likes ? `<span>👍 ${c.likes}</span>` : ''}
        ${date ? `<span>📅 ${date}</span>` : ''}
      </div>
    </div>`;
  }).join('');
}

/* Insights */
function renderInsights() {
  const total = state.results.length;
  if (!total) return;
  const pos = state.results.filter(r=>r.sentiment==='Positive').length;
  const neg = state.results.filter(r=>r.sentiment==='Negative').length;
  const neu = state.results.filter(r=>r.sentiment==='Neutral').length;
  const pPct = Math.round(pos/total*100);
  const negPct = Math.round(neg/total*100);

  const mood = pPct > 60 ? '🟢 Very Positive' : pPct > 40 ? '🟡 Mostly Positive' : negPct > 40 ? '🔴 Largely Negative' : '⚪ Mixed Audience';
  const engagement = total > 100 ? 'High' : total > 50 ? 'Moderate' : 'Low';
  const reception = pPct >= 70 ? 'Excellent' : pPct >= 50 ? 'Good' : negPct >= 50 ? 'Poor' : 'Fair';
  const satisfaction = pPct >= 60 ? `${pPct}% — High` : pPct >= 40 ? `${pPct}% — Moderate` : `${pPct}% — Low`;
  const toxicity = negPct <= 10 ? 'Very Low' : negPct <= 25 ? 'Low' : negPct <= 40 ? 'Moderate' : 'High';
  const trend = pos > neu && pos > neg ? '📈 Positive momentum' : neg > pos ? '📉 Needs attention' : '📊 Balanced audience';
  const rec = pPct >= 60 ? 'Content is performing well. Encourage more audience interaction.' : negPct >= 40 ? 'Address negative comments and consider adjusting content strategy.' : 'Maintain current approach and monitor for shifts in sentiment.';

  setText('ins-mood',           mood);
  setText('ins-engagement',     engagement);
  setText('ins-reception',      reception);
  setText('ins-satisfaction',   satisfaction);
  setText('ins-toxicity',       toxicity);
  setText('ins-trend',          trend);
  setText('ins-recommendation', rec);
}

/* AI Summary */
async function renderAISummary() {
  const el = document.getElementById('ai-summary');
  if (!el) return;
  const total = state.results.length;
  const pos = state.results.filter(r=>r.sentiment==='Positive').length;
  const neg = state.results.filter(r=>r.sentiment==='Negative').length;
  const neu = state.results.filter(r=>r.sentiment==='Neutral').length;
  const pPct = total ? Math.round(pos/total*100) : 0;
  const negPct = total ? Math.round(neg/total*100) : 0;
  const title = state.videoDetails?.title || 'this video';

  if (typeof CONFIG !== 'undefined' && CONFIG.GEMINI_API_KEY && !CONFIG.GEMINI_API_KEY.startsWith('AQ')) {
    el.textContent = 'Generating AI summary with Gemini…';
    try {
      const prompt = `You are a YouTube analytics expert. Based on these stats, write a 2-3 sentence audience summary: Video "${title}" analyzed ${total} comments. ${pPct}% Positive, ${Math.round(neu/total*100)}% Neutral, ${negPct}% Negative. Be concise, insightful, and professional.`;
      const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${CONFIG.GEMINI_API_KEY}`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ contents:[{ parts:[{ text: prompt }] }] })
      });
      if (res.ok) {
        const data = await res.json();
        const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (text) { el.textContent = text; return; }
      }
    } catch {}
  }

  // Fallback summary
  const dom = pPct > negPct && pPct > Math.round(neu/total*100) ? 'positive' : negPct > pPct ? 'negative' : 'mixed';
  el.textContent = `Analysis of ${total} comments from "${title}" reveals a predominantly ${dom} audience sentiment (${pPct}% positive, ${negPct}% negative, ${Math.round(neu/total*100)}% neutral). ${pPct >= 60 ? 'Viewers are highly engaged and appreciate the content quality.' : negPct >= 40 ? 'The video has received significant criticism — consider reviewing feedback themes.' : 'The audience response is balanced, with diverse opinions expressed in the comments.'}`;
}

/* ─── EXPORT ────────────────────────────────────────────────── */
function downloadPDF() {
  showToast('Generating PDF…', 'info');
  const el = document.getElementById('dash-printable');
  if (!el || typeof html2pdf === 'undefined') {
    showToast('PDF library not loaded', 'error'); return;
  }
  html2pdf().set({
    margin:[10,10], filename:`VibeCheck-Report-${state.videoDetails?.id || 'export'}.pdf`,
    image:{type:'jpeg',quality:.9},
    html2canvas:{scale:1.5,backgroundColor:'#150E22',useCORS:true},
    jsPDF:{unit:'mm',format:'a4',orientation:'portrait'},
    pagebreak:{mode:['avoid-all','css']}
  }).from(el).save().then(() => showToast('PDF downloaded', 'success'));
}

function downloadCSV() {
  if (!state.results.length) { showToast('No data to export', 'error'); return; }
  const header = ['Author','Comment','Sentiment','Confidence','Likes','Date'];
  const rows = state.results.map(r => [
    csvEscape(r.author || ''),
    csvEscape(r.text.replace(/<[^>]*>/g,'') || ''),
    r.sentiment,
    r.score != null ? Math.round(r.score*100)+'%' : '',
    r.likes || 0,
    r.publishedAt ? new Date(r.publishedAt).toLocaleDateString() : ''
  ]);
  const csv = [header, ...rows].map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `VibeCheck-${state.videoDetails?.id || 'export'}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
  showToast('CSV downloaded', 'success');
}

function csvEscape(s) {
  return `"${String(s).replace(/"/g,'""').slice(0,500)}"`;
}

/* ─── MOBILE NAV ────────────────────────────────────────────── */
function toggleMobileNav() {
  const mob = document.getElementById('nav-mobile');
  if (mob) mob.classList.toggle('open');
}

/* ─── SCROLL BEHAVIORS ──────────────────────────────────────── */
function initScrollReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!els.length) return;
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target) } });
  }, { threshold:0.12, rootMargin:'0px 0px -40px 0px' });
  els.forEach(el => io.observe(el));
}

function initNavScroll() {
  const nav = document.getElementById('main-nav');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 20);
  }, {passive:true});
}

/* ─── UTILS ─────────────────────────────────────────────────── */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function countUp(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  const duration = 1200;
  const start = performance.now();
  const update = (now) => {
    const t = Math.min((now-start)/duration, 1);
    const ease = 1 - Math.pow(1-t, 3);
    el.textContent = Math.round(ease * target);
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function formatNum(n) {
  if (!n) return '0';
  if (n >= 1e9) return (n/1e9).toFixed(1)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return String(n);
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showToast(msg, type='info') {
  const root = document.getElementById('toast-root');
  if (!root) return;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-dot"></span><span class="toast-msg">${escHtml(msg)}</span>`;
  root.appendChild(el);
  setTimeout(() => { el.style.opacity='0'; el.style.transform='translateX(20px)'; el.style.transition='all .3s ease'; setTimeout(()=>el.remove(), 300) }, 4000);
}

/* ─── URL PARAMS (deep link support) ───────────────────────── */
function checkURLParams() {
  const params = new URLSearchParams(window.location.search);
  const url = params.get('url');
  if (url) {
    const input = document.getElementById('hero-url');
    if (input) { input.value = url; startAnalysis('hero-url') }
  }
}

/* ─── KEYBOARD ──────────────────────────────────────────────── */
function initKeyboard() {
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const landing = document.getElementById('landing-page');
      if (landing && landing.style.display !== 'none') {
        const hero = document.getElementById('hero-url');
        const tryInput = document.getElementById('try-url');
        if (document.activeElement === hero) startAnalysis('hero-url');
        if (document.activeElement === tryInput) startAnalysis('try-url');
      }
    }
    if (e.key === 'Escape') {
      const loading = document.getElementById('loading-screen');
      if (loading && loading.style.display !== 'none') goToLanding();
    }
  });
}

/* ─── NAV SMOOTH LINK ───────────────────────────────────────── */
function initNavLinks() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior:'smooth' });
        const mob = document.getElementById('nav-mobile');
        if (mob) mob.classList.remove('open');
      }
    });
  });
}

/* ─── INIT ──────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initScrollReveal();
  initNavScroll();
  initNavLinks();
  initKeyboard();
  checkURLParams();
});
