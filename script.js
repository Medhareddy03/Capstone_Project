/* global FaceDetector, FaceLandmarker, FilesetResolver, ImageClassifier, ImageEmbedder */
// ═══════════════════════════════════════════════════════════════
//      Core Logic
// ════════════════════════════════════════════════════════════════

// ─── STATE ───────────────────────────────────────────────────────────────────
const S = {
    mode: 'upload',
    stream: null,
    rafId: null,

    // ── MediaPipe task instances ──
    faceDetector: null,   // BlazeFace short-range (fast guard, every frame)
    faceLandmarker: null,   // 478 landmarks + 52 blendshapes
    imageClassifier: null,   // EfficientNet Lite0 — lazy loaded on first deep analysis
    imageEmbedder: null,   // MobileNet V3 Small — lazy loaded on first deep analysis

    // ── Running mode tracking (IMAGE | VIDEO) ──
    detectorMode: null,
    landmarkerMode: null,

    // ── Analysis state ──
    faceFound: false,
    imgData: null,   // ImageData from last analyzed frame
    canvasImg: null,   // HTMLImageElement for static upload
    lastBlendshapes: [],     // 52 blendshape categories from FaceLandmarker
    lastEmbedding: null,   // Float32Array from ImageEmbedder
    lastClassification: null,// Top classification labels

    // ── Mesh style: 'full' | 'contour' | 'minimal' | 'off' ──
    meshStyle: 'full',
};

// ─── DOM refs ────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const canvas = $('output-canvas');
const video = $('video');
const ctx = canvas.getContext('2d');

// ─── STATUS ──────────────────────────────────────────────────────────────────
function setStatus(state, msg) {
    const dot = $('status-dot');
    const txt = $('status-text');
    dot.className = 'dot ' + state;
    txt.textContent = msg;
}

// ─── MODE SWITCH ─────────────────────────────────────────────────────────────
async function switchMode(m) {
    if (m === S.mode) return;
    S.mode = m;
    $('tab-upload').className = 'mode-tab' + (m === 'upload' ? ' active' : '');
    $('tab-webcam').className = 'mode-tab' + (m === 'webcam' ? ' active' : '');
    if (m === 'webcam') {
        $('drop-zone').classList.add('hidden');
        startCam();
    } else {
        stopCam();
        await Promise.all([ensureDetectorMode('IMAGE'), ensureLandmarkerMode('IMAGE')]);
        $('drop-zone').classList.remove('hidden');
        video.style.display = 'none';
        canvas.style.display = 'none';
        $('analyze-btn').disabled = true;
    }
}

// ─── MEDIAPIPE TASKS VISION INIT ─────────────────────────────────────────────
// Model URLs
const WASM_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models';

const MODELS = {
    detector: `${MODEL_BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
    landmarker: `${MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
    classifier: `${MODEL_BASE}/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite`,
    embedder: `${MODEL_BASE}/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite`,
};

// ─── Backend API
const BACKEND_URL = 'http://localhost:8000';

async function waitForMPTasks() {
    if (window.FaceLandmarker) return;
    await new Promise(r => window.addEventListener('mp-tasks-ready', r, { once: true }));
}

// Build one shared FilesetResolver (WASM downloaded once, cached for all tasks)
let _visionPromise = null;
function getVision() {
    if (!_visionPromise) _visionPromise = FilesetResolver.forVisionTasks(WASM_CDN);
    return _visionPromise;
}

async function makeDetector(mode) {
    const v = await getVision();
    return FaceDetector.createFromOptions(v, {
        baseOptions: { modelAssetPath: MODELS.detector, delegate: 'GPU' },
        runningMode: mode,
        minDetectionConfidence: 0.5,
        minSuppressionThreshold: 0.3,
    });
}

async function makeLandmarker(mode) {
    const v = await getVision();
    return FaceLandmarker.createFromOptions(v, {
        baseOptions: { modelAssetPath: MODELS.landmarker, delegate: 'GPU' },
        runningMode: mode,
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFaceBlendshapes: true,   // 52 expression coefficients
        outputFacialTransformationMatrixes: false,
    });
}

async function initAllTasks() {
    try {
        setStatus('processing', 'Downloading MediaPipe WASM + models (first load only)…');
        await waitForMPTasks();

        // Init FaceDetector + FaceLandmarker in parallel — both start in IMAGE mode
        [S.faceDetector, S.faceLandmarker] = await Promise.all([
            makeDetector('IMAGE'),
            makeLandmarker('IMAGE'),
        ]);
        S.detectorMode = 'IMAGE';
        S.landmarkerMode = 'IMAGE';

        setStatus('ready', 'All vision tasks ready — upload a portrait or start webcam');
    } catch (e) {
        setStatus('error', 'Init failed: ' + e.message);
        console.error(e);
    }
}

// ── Lazy-load Classifier + Embedder on first deep-analysis click ──
async function ensureClassifierEmbedder() {
    if (S.imageClassifier && S.imageEmbedder) return;
    const v = await getVision();
    const tasks = [];
    if (!S.imageClassifier) tasks.push(
        ImageClassifier.createFromOptions(v, {
            baseOptions: { modelAssetPath: MODELS.classifier, delegate: 'GPU' },
            runningMode: 'IMAGE',
            maxResults: 5,
            scoreThreshold: 0.05,
        }).then(t => { S.imageClassifier = t; })
    );
    if (!S.imageEmbedder) tasks.push(
        ImageEmbedder.createFromOptions(v, {
            baseOptions: { modelAssetPath: MODELS.embedder, delegate: 'GPU' },
            runningMode: 'IMAGE',
            quantize: false,
        }).then(t => { S.imageEmbedder = t; })
    );
    await Promise.all(tasks);
}

// ── Running-mode switchers ──
async function ensureDetectorMode(mode) {
    if (!S.faceDetector || S.detectorMode === mode) return;
    await S.faceDetector.setOptions({ runningMode: mode });
    S.detectorMode = mode;
}
async function ensureLandmarkerMode(mode) {
    if (!S.faceLandmarker || S.landmarkerMode === mode) return;
    await S.faceLandmarker.setOptions({ runningMode: mode });
    S.landmarkerMode = mode;
}

// ─── IMAGE UPLOAD ─────────────────────────────────────────────────────────────
function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
        S.canvasImg = img;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);

        $('drop-zone').classList.add('hidden');
        canvas.style.display = 'block';

        setStatus('processing', 'Analyzing…');
        await runFaceLandmarker(img);
    };
    img.src = url;
}

// Drag & drop
const wrap = $('canvas-wrap');
wrap.addEventListener('dragover', e => e.preventDefault());
wrap.addEventListener('drop', e => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) {
        handleFile({ target: { files: [f] } });
    }
});

// ─── WEBCAM ──────────────────────────────────────────────────────────────────
async function startCam() {
    try {
        S.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        video.srcObject = S.stream;
        video.style.display = 'block';
        $('cam-controls').classList.add('visible');
        video.onloadeddata = async () => {
            // Switch both tasks to VIDEO mode before starting the loop
            await Promise.all([ensureDetectorMode('VIDEO'), ensureLandmarkerMode('VIDEO')]);
            setStatus('ready', 'Webcam active — analyzing live');
            loopCam();
        };
    } catch (err) {
        setStatus('error', 'Camera: ' + err.message);
        switchMode('upload');
    }
}

function stopCam() {
    if (S.stream) { S.stream.getTracks().forEach(t => t.stop()); S.stream = null; }
    if (S.rafId) { cancelAnimationFrame(S.rafId); S.rafId = null; }
    $('cam-controls').classList.remove('visible');
    video.style.display = 'none';
}

function loopCam() {
    if (!S.stream || !S.faceDetector || !S.faceLandmarker) {
        S.rafId = requestAnimationFrame(loopCam);
        return;
    }
    if (video.readyState >= 2) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0);
        ctx.restore();
        S.imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        updateColorTemp(S.imgData);
        canvas.style.display = 'block';

        const ts = performance.now();

        // ── Stage 1: FaceDetector guard (BlazeFace, ~2 ms) ──
        const det = S.faceDetector.detectForVideo(video, ts);
        updateDetectionUI(det, canvas.width, canvas.height);

        if (det.detections && det.detections.length > 0) {
            // ── Stage 2: FaceLandmarker (only when face confirmed) ──
            const lmResult = S.faceLandmarker.detectForVideo(video, ts);
            onFaceResults(lmResult, det);
        } else {
            S.faceFound = false;
            setStatus('processing', 'No face detected — please look at the camera');
        }
    }
    S.rafId = requestAnimationFrame(loopCam);
}

function captureFrame() {
    $('analyze-btn').disabled = false;
    setStatus('ready', 'Frame captured — ready for deep analysis');
}

// ─── FACE PIPELINE (static image) ────────────────────────────────────────────
async function runFaceLandmarker(source) {
    if (!S.faceDetector || !S.faceLandmarker) return;
    await Promise.all([ensureDetectorMode('IMAGE'), ensureLandmarkerMode('IMAGE')]);
    S.imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    updateColorTemp(S.imgData);

    // Stage 1: detect
    const det = S.faceDetector.detect(source);
    updateDetectionUI(det, canvas.width, canvas.height);

    if (!det.detections || det.detections.length === 0) {
        setStatus('error', 'No face detected — try another image');
        return;
    }
    // Stage 2: landmark
    const lmResult = S.faceLandmarker.detect(source);
    onFaceResults(lmResult, det);
}

// ─── DISTANCE RANGE SETTINGS ─────────────────────────────────────────────────
// IPD expressed as % of frame width.
// ~10-28% covers typical "portrait selfie" range comfortably.
const DIST = { min: 10, max: 28 };

function toggleSettings() {
    const body = $('settings-body');
    const chev = $('settings-chevron');
    const open = body.classList.toggle('visible');
    chev.classList.toggle('open', open);
}

function onDistSlider() {
    let mn = parseInt($('dist-min').value);
    let mx = parseInt($('dist-max').value);
    // Prevent overlap: keep at least 3% gap
    if (mn >= mx - 3) { mn = mx - 3; $('dist-min').value = mn; }
    DIST.min = mn; DIST.max = mx;
    $('dist-min-val').textContent = mn + '%';
    $('dist-max-val').textContent = mx + '%';
    updateDistZone();
}

// Map IPD% (5–40 range) → 0–100% position on the track bar
function ipdToTrackPct(ipd) {
    return Math.min(100, Math.max(0, ((ipd - 5) / (40 - 5)) * 100));
}

function updateDistZone() {
    const zl = ipdToTrackPct(DIST.min);
    const zr = ipdToTrackPct(DIST.max);
    $('dist-zone').style.left = zl + '%';
    $('dist-zone').style.width = (zr - zl) + '%';
}

// Call once at boot
updateDistZone();

// ─── DISTANCE CALCULATION ────────────────────────────────────────────────────
// Uses iris centre landmarks 468 (left) and 473 (right).
// In the new Tasks Vision FaceLandmarker these are ALWAYS present —
// there is no separate refineLandmarks flag. Falls back to outer eye
// corners 33/263 only if somehow absent.
function calcFaceIPD(lm, W, H) {
    const lEye = lm[468] || lm[33];
    const rEye = lm[473] || lm[263];
    const dx = (rEye.x - lEye.x) * W;
    const dy = (rEye.y - lEye.y) * H;
    return (Math.sqrt(dx * dx + dy * dy) / W) * 100; // % of frame width
}

// ─── DISTANCE UI UPDATE ──────────────────────────────────────────────────────
function updateDistanceUI(ipd) {
    const pinPct = ipdToTrackPct(ipd);
    const pin = $('dist-pin');
    const stat = $('dist-status');
    const valEl = $('val-dist');
    const subEl = $('sub-dist');

    pin.style.left = pinPct + '%';

    let label, zone, statusText, statusClass;

    if (ipd < DIST.min) {
        label = 'Too Far'; zone = 'too-far';
        statusText = 'Move closer to camera'; statusClass = 'far';
    } else if (ipd > DIST.max) {
        label = 'Too Close'; zone = 'too-close';
        statusText = 'Step back a little'; statusClass = 'near';
    } else {
        label = 'Good Distance'; zone = 'in-range';
        statusText = 'Distance is within range'; statusClass = 'ok';
    }

    pin.className = 'dist-pin ' + zone;
    valEl.textContent = label;
    valEl.style.color = zone === 'in-range' ? 'var(--sage)' : zone === 'too-close' ? 'var(--rose)' : 'var(--gold)';
    subEl.textContent = `IPD ${ipd.toFixed(1)}% of frame · range ${DIST.min}–${DIST.max}%`;
    stat.textContent = statusText;
    stat.className = 'dist-status ' + statusClass;
    $('card-dist').classList.add('lit');
}

// ─── DETECTION UI ────────────────────────────────────────────────────────────
function updateDetectionUI(det, W, H) {
    if (!det.detections || det.detections.length === 0) return;
    const d = det.detections[0];
    const score = Math.round((d.categories?.[0]?.score ?? 0) * 100);
    const bb = d.boundingBox;
    const circ = 2 * Math.PI * 18; // circumference of r=18 arc

    $('val-detect').textContent = `${score}% confidence`;
    $('det-arc-pct').textContent = score + '%';
    $('det-arc').setAttribute('stroke-dasharray',
        `${(score / 100) * circ} ${circ}`);
    $('det-box-w').textContent = bb ? Math.round(bb.width) + 'px' : '—';
    $('det-box-h').textContent = bb ? Math.round(bb.height) + 'px' : '—';
    $('det-count').textContent = det.detections.length;
    $('card-detect').classList.add('lit');

    // Draw bounding box on canvas
    if (bb && S.meshStyle !== 'off') {
        ctx.strokeStyle = 'rgba(196,154,108,0.45)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(bb.originX, bb.originY, bb.width, bb.height);
        ctx.setLineDash([]);
    }
}

// ─── BLENDSHAPE UI ───────────────────────────────────────────────────────────
// Top expressions to surface — maps raw blendshape name → readable label
const BLEND_DISPLAY = [
    ['mouthSmileLeft', 'Smile L'],
    ['mouthSmileRight', 'Smile R'],
    ['eyeBlinkLeft', 'Blink L'],
    ['eyeBlinkRight', 'Blink R'],
    ['browInnerUp', 'Brow raise'],
    ['jawOpen', 'Jaw open'],
    ['cheekPuff', 'Cheek puff'],
    ['mouthPucker', 'Pucker'],
    ['eyeSquintLeft', 'Squint L'],
    ['eyeSquintRight', 'Squint R'],
];

function updateBlendshapeUI(blendshapes) {
    if (!blendshapes || !blendshapes.length) return;
    S.lastBlendshapes = blendshapes;

    // Build lookup map
    const map = {};
    blendshapes.forEach(b => { map[b.categoryName] = b.score; });

    const dominant = blendshapes
        .slice()
        .sort((a, b) => b.score - a.score)
        .slice(0, 3)
        .filter(b => b.score > 0.08)
        .map(b => {
            const display = BLEND_DISPLAY.find(([k]) => k === b.categoryName);
            return display ? display[1] : b.categoryName.replace(/([A-Z])/g, ' $1').trim();
        });

    $('val-blend').textContent = dominant.length
        ? dominant.join(' · ')
        : 'Neutral expression';

    const list = $('blend-list');
    list.innerHTML = BLEND_DISPLAY.map(([key, label]) => {
        const score = map[key] ?? 0;
        const pct = Math.round(score * 100);
        return `<div class="blend-row">
      <span class="blend-name">${label}</span>
      <div class="blend-bg"><div class="blend-fill" style="width:${pct}%"></div></div>
      <span class="blend-score">${pct}%</span>
    </div>`;
    }).join('');

    $('card-blend').classList.add('lit');
}

// ─── RESULTS HANDLER ─────────────────────────────────────────────────────────
// result  = FaceLandmarker result { faceLandmarks, faceBlendshapes }
// det     = FaceDetector result   { detections }
function onFaceResults(result, det) {
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
        if (S.mode === 'upload') setStatus('error', 'No face landmarks — try another image');
        S.faceFound = false;
        return;
    }

    const lm = result.faceLandmarks[0];
    const W = canvas.width, H = canvas.height;

    // ── Distance check ──
    const ipd = calcFaceIPD(lm, W, H);
    updateDistanceUI(ipd);
    const inRange = (ipd >= DIST.min && ipd <= DIST.max);

    if (S.mode === 'webcam' && !inRange) {
        drawOverlay(lm, W, H);
        const hint = ipd < DIST.min ? 'Move closer' : 'Step back';
        setStatus('processing', `Out of range · ${hint} · IPD ${ipd.toFixed(1)}%`);
        return;
    }

    if (S.mode === 'upload' && S.canvasImg) ctx.drawImage(S.canvasImg, 0, 0, W, H);

    S.imgData = ctx.getImageData(0, 0, W, H);
    S.faceFound = true;

    // ── Pixel-based analyses ──
    const skin = analyzeSkinTone(S.imgData, lm, W, H);
    const shape = analyzeFaceShape(lm, W, H);
    const orns = detectOrnaments(S.imgData, lm, W, H);

    updateSkinUI(skin);
    updateFaceUI(shape);
    updateOrnUI(orns);

    // ── Blendshapes ──
    if (result.faceBlendshapes?.[0]?.categories) {
        updateBlendshapeUI(result.faceBlendshapes[0].categories);
    }

    // ── Redraw bounding box if we have det ──
    if (det) updateDetectionUI(det, W, H);

    // ── Mesh overlay ──
    drawOverlay(lm, W, H);

    $('analyze-btn').disabled = false;
    setStatus('ready', `Face in range · ${lm.length} landmarks · IPD ${ipd.toFixed(1)}%`);
}

// ─── COLOR SPACE MATH ─────────────────────────────────────────────────────────
function sRGBtoLinear(c) {
    c /= 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function rgbToLab(r, g, b) {
    const rl = sRGBtoLinear(r), gl = sRGBtoLinear(g), bl = sRGBtoLinear(b);
    // → XYZ (D65)
    let X = rl * 41.239 + gl * 35.758 + bl * 18.048;
    let Y = rl * 21.267 + gl * 71.515 + bl * 7.218;
    let Z = rl * 1.933 + gl * 11.919 + bl * 95.053;
    // Normalize
    const labF = (t) => t > 8.856 ? Math.cbrt(t / 100) : t / 902.9 + 0.1379;
    // Actually correct D65 white point: Xn=95.047, Yn=100, Zn=108.883
    const f = (t, n) => { const v = t / n; return v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116; };
    const fx = f(X, 95.047), fy = f(Y, 100.0), fz = f(Z, 108.883);
    return { L: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
}

// ─── PIXEL SAMPLING ──────────────────────────────────────────────────────────
// Skin landmark indices (cheeks, nose, forehead — avoiding eyes/lips/brows)
const SKIN_LM = [1, 4, 5, 6, 10, 36, 50, 100, 101, 116, 117, 123, 205, 234,
    266, 280, 329, 330, 345, 346, 425, 338, 297, 332, 284, 251];

function sampleArea(imgData, lm, W, H, r = 4) {
    const cx = Math.round(lm.x * W);
    const cy = Math.round(lm.y * H);
    let rv = 0, gv = 0, bv = 0, n = 0;
    for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
            const px = cx + dx, py = cy + dy;
            if (px >= 0 && px < W && py >= 0 && py < H) {
                const i = (py * W + px) * 4;
                rv += imgData.data[i]; gv += imgData.data[i + 1]; bv += imgData.data[i + 2];
                n++;
            }
        }
    }
    return n > 0 ? { r: rv / n, g: gv / n, b: bv / n } : null;
}

function sampleRegion(imgData, cx, cy, w, h, W, H) {
    const samples = [];
    const step = Math.max(1, Math.round(Math.min(w, h) / 8));
    for (let dy = 0; dy < h; dy += step) {
        for (let dx = 0; dx < w; dx += step) {
            const px = Math.round(cx - w / 2 + dx);
            const py = Math.round(cy + dy);
            if (px >= 0 && px < W && py >= 0 && py < H) {
                const i = (py * W + px) * 4;
                const bright = (imgData.data[i] + imgData.data[i + 1] + imgData.data[i + 2]) / 3;
                const sat = Math.max(imgData.data[i], imgData.data[i + 1], imgData.data[i + 2])
                    - Math.min(imgData.data[i], imgData.data[i + 1], imgData.data[i + 2]);
                samples.push({ bright, sat, r: imgData.data[i], g: imgData.data[i + 1], b: imgData.data[i + 2] });
            }
        }
    }
    if (!samples.length) return null;
    const avg = samples.reduce((a, s) => a + s.bright, 0) / samples.length;
    const variance = samples.reduce((a, s) => a + (s.bright - avg) ** 2, 0) / samples.length;
    return {
        variance,
        avgBright: avg,
        maxBright: Math.max(...samples.map(s => s.bright)),
        avgSat: samples.reduce((a, s) => a + s.sat, 0) / samples.length
    };
}

// ─── SKIN TONE ANALYSIS ───────────────────────────────────────────────────────
function analyzeSkinTone(imgData, lm, W, H) {
    let rS = 0, gS = 0, bS = 0, n = 0;
    for (const idx of SKIN_LM) {
        if (!lm[idx]) continue;
        const px = sampleArea(imgData, lm[idx], W, H);
        if (px) { rS += px.r; gS += px.g; bS += px.b; n++; }
    }
    if (n === 0) return null;

    const r = rS / n, g = gS / n, b = bS / n;
    const lab = rgbToLab(r, g, b);
    const ita = Math.atan2(lab.L - 50, lab.b) * (180 / Math.PI);

    let fitzpatrick, label;
    if (ita > 55) { fitzpatrick = 'Type I'; label = 'Very Light'; }
    else if (ita > 41) { fitzpatrick = 'Type II'; label = 'Light'; }
    else if (ita > 28) { fitzpatrick = 'Type III'; label = 'Light Medium'; }
    else if (ita > 10) { fitzpatrick = 'Type IV'; label = 'Medium Dark'; }
    else if (ita > -30) { fitzpatrick = 'Type V'; label = 'Dark'; }
    else { fitzpatrick = 'Type VI'; label = 'Deep Dark'; }

    const warmBias = r - b;
    const greenBias = g - (r + b) / 2;
    let undertone;
    if (warmBias > 28) undertone = 'Warm · Golden';
    else if (warmBias < 5 && (r - g) < 8) undertone = 'Cool · Pink-Rose';
    else if (greenBias > 8) undertone = 'Neutral-Olive';
    else undertone = 'Neutral';

    // Confidence: how well defined the skin region is
    const conf = Math.round(Math.min(95, 50 + Math.abs(ita) * 0.6));

    return { fitzpatrick, label, undertone, rgb: { r, g, b }, ita, conf };
}

function updateSkinUI(t) {
    if (!t) return;
    $('val-skin').textContent = t.label;
    $('sub-skin').textContent = `${t.fitzpatrick} · ${t.undertone}`;
    $('swatch-skin').style.background = `rgb(${Math.round(t.rgb.r)},${Math.round(t.rgb.g)},${Math.round(t.rgb.b)})`;
    $('conf-skin').style.width = t.conf + '%';
    $('card-skin').classList.add('lit');
}

// ─── COLOR TEMPERATURE ────────────────────────────────────────────────────────
function updateColorTemp(imgData) {
    let r = 0, g = 0, b = 0;
    const step = 20; // sample every Nth pixel (performance)
    let n = 0;
    for (let i = 0; i < imgData.data.length; i += step * 4) {
        r += imgData.data[i]; g += imgData.data[i + 1]; b += imgData.data[i + 2];
        n++;
    }
    r /= n; g /= n; b /= n;

    const rtob = r / (b + 1);
    let label, sub, pin;
    if (rtob > 1.6) { label = 'Candlelight'; sub = '~2700K · Very Warm'; pin = 90; }
    else if (rtob > 1.3) { label = 'Warm White'; sub = '~3200K · Warm'; pin = 72; }
    else if (rtob > 1.1) { label = 'Natural Warm'; sub = '~4500K · Neutral+'; pin = 58; }
    else if (rtob > 0.9) { label = 'Daylight'; sub = '~5500K · Neutral'; pin = 44; }
    else if (rtob > 0.75) { label = 'Overcast'; sub = '~7000K · Cool'; pin = 28; }
    else { label = 'Cool Blue'; sub = '~9000K+ · Cold'; pin = 12; }

    $('val-temp').textContent = label;
    $('sub-temp').textContent = sub;
    $('temp-pin').style.left = pin + '%';
    $('card-temp').classList.add('lit');
}

// ─── FACE SHAPE ───────────────────────────────────────────────────────────────
function analyzeFaceShape(lm, W, H) {
    const pt = (i) => ({ x: lm[i].x * W, y: lm[i].y * H });
    const d = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

    const cheekW = d(pt(234), pt(454));
    const jawW = d(pt(172), pt(397));
    const faceH = d(pt(10), pt(152));
    const foreW = d(pt(54), pt(284));
    const chinW = d(pt(58), pt(288));

    const hRatio = faceH / (cheekW || 1);
    const jRatio = jawW / (cheekW || 1);
    const fRatio = foreW / (cheekW || 1);
    const cRatio = chinW / (jawW || 1);

    let shape, detail;
    if (hRatio < 1.05 && jRatio > 0.82) { shape = 'Round'; detail = 'Equal width & height, soft curves'; }
    else if (jRatio > 0.87 && Math.abs(fRatio - jRatio) < 0.08) { shape = 'Square'; detail = 'Angular, strong jaw, even width'; }
    else if (fRatio - jRatio > 0.20) { shape = 'Heart'; detail = 'Wide forehead, delicate chin'; }
    else if (jRatio < fRatio - 0.15 && cheekW > foreW) { shape = 'Diamond'; detail = 'High cheekbones, narrow jaw'; }
    else if (hRatio > 1.40 && jRatio < 0.78) { shape = 'Oblong'; detail = 'Long, slender, high forehead'; }
    else { shape = 'Oval'; detail = 'Balanced, gently tapered'; }

    return { shape, detail, metrics: { hRatio, jRatio, fRatio } };
}

function updateFaceUI(f) {
    if (!f) return;
    $('val-face').textContent = f.shape;
    $('sub-face').textContent = f.detail;

    const grid = $('measure-grid');
    grid.style.display = 'grid';
    grid.innerHTML = [
        ['H/W Ratio', f.metrics.hRatio.toFixed(2)],
        ['Jaw Ratio', f.metrics.jRatio.toFixed(2)],
        ['Forehead', f.metrics.fRatio.toFixed(2)],
        ['Shape', f.shape],
    ].map(([k, v]) => `<div class="measure-item"><div class="measure-key">${k}</div><div class="measure-val">${v}</div></div>`).join('');

    $('card-face').classList.add('lit');
}

// ─── ORNAMENT DETECTION ───────────────────────────────────────────────────────
function detectOrnaments(imgData, lm, W, H) {
    const pt = (i) => ({ x: lm[i].x * W, y: lm[i].y * H });
    const found = [];

    // GLASSES: analyze eye-bridge + frame region for dark linear structures
    const bridge = pt(168);
    const lEyeInner = pt(133), rEyeInner = pt(362);
    const eyeSpan = Math.abs(rEyeInner.x - lEyeInner.x);

    const leftFrame = sampleRegion(imgData, lEyeInner.x, lEyeInner.y - eyeSpan * 0.2, eyeSpan * 0.5, eyeSpan * 0.35, W, H);
    const rightFrame = sampleRegion(imgData, rEyeInner.x, rEyeInner.y - eyeSpan * 0.2, eyeSpan * 0.5, eyeSpan * 0.35, W, H);

    if (leftFrame && rightFrame) {
        const darkFrames = (leftFrame.avgBright < 90 && rightFrame.avgBright < 90);
        const brightFrames = (leftFrame.variance > 400 && rightFrame.variance > 400 && leftFrame.maxBright > 200);
        if (darkFrames || brightFrames) found.push('Glasses');
    }

    // EARRINGS: regions lateral to ear landmarks (234 = left ear, 454 = right ear)
    const lEar = pt(234), rEar = pt(454);
    const earW = Math.max(30, cheekSize(lm, W, H) * 0.15);
    const earH = earW * 1.8;

    const lEarReg = sampleRegion(imgData, lEar.x - earW * 0.6, lEar.y - earH * 0.3, earW, earH, W, H);
    const rEarReg = sampleRegion(imgData, rEar.x - earW * 0.4, rEar.y - earH * 0.3, earW, earH, W, H);

    const earringSignal = (reg) => reg && reg.variance > 600 && reg.maxBright > 155 && reg.avgSat > 15;
    if (earringSignal(lEarReg) || earringSignal(rEarReg)) found.push('Earrings');

    // NECKLACE: below chin (landmark 152)
    const chin = pt(152);
    const neckW = earW * 3;
    const neck = sampleRegion(imgData, chin.x, chin.y + earW * 0.4, neckW, earW * 1.2, W, H);
    if (neck && neck.variance > 800 && neck.maxBright > 160) found.push('Necklace');

    // BINDI: small region between brows on forehead (landmark 9 = glabella)
    const glabella = pt(9);
    const bindi = sampleRegion(imgData, glabella.x, glabella.y - 2, 18, 18, W, H);
    if (bindi && bindi.avgSat > 75 && bindi.variance > 350) found.push('Bindi / Marking');

    return found;
}

function cheekSize(lm, W, H) {
    const l = { x: lm[234].x * W, y: lm[234].y * H };
    const r = { x: lm[454].x * W, y: lm[454].y * H };
    return Math.sqrt((l.x - r.x) ** 2 + (l.y - r.y) ** 2);
}

function updateOrnUI(orns) {
    $('val-orn').textContent = orns.length ? orns.length + ' detected' : 'None visible';
    $('tag-list').innerHTML = orns.map(o => `<span class="tag">${o}</span>`).join('');
    $('card-orn').classList.add('lit');
}

// ─── MESH STYLE TOGGLE ───────────────────────────────────────────────────────
function setMesh(style) {
    S.meshStyle = style;
    ['full', 'contour', 'minimal', 'off'].forEach(s => {
        $('pill-' + s).classList.toggle('active', s === style);
    });
}

// ─── OVERLAY DRAWING ─────────────────────────────────────────────────────────
// Uses FaceLandmarker static connection constants — provided by the Tasks Vision
// API directly on the class, no manual index arrays needed.
//
// FaceLandmarker.FACE_LANDMARKS_TESSELATION   — ~1404 connections, full mesh
// FaceLandmarker.FACE_LANDMARKS_FACE_OVAL     — face outline contour
// FaceLandmarker.FACE_LANDMARKS_LEFT_EYE      — left eye contour
// FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE     — right eye contour
// FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW  — left brow
// FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW — right brow
// FaceLandmarker.FACE_LANDMARKS_LIPS          — full lip outline
// FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS     — left iris ring
// FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS    — right iris ring
// Each constant is an array of {start: number, end: number} index pairs.

function drawConnections(lm, connections, W, H) {
    connections.forEach(({ start, end }) => {
        const s = lm[start], e = lm[end];
        if (!s || !e) return;
        ctx.beginPath();
        ctx.moveTo(s.x * W, s.y * H);
        ctx.lineTo(e.x * W, e.y * H);
        ctx.stroke();
    });
}

function drawOverlay(lm, W, H) {
    if (S.meshStyle === 'off') return;

    ctx.save();

    if (S.meshStyle === 'full') {
        // ── Full tesselation mesh ── (~1404 edges, very detailed)
        ctx.strokeStyle = 'rgba(196,154,108,0.10)';
        ctx.lineWidth = 0.5;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, W, H);
    }

    if (S.meshStyle === 'full' || S.meshStyle === 'contour') {
        // ── Face oval ──
        ctx.strokeStyle = 'rgba(196,154,108,0.45)';
        ctx.lineWidth = 1;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, W, H);

        // ── Eyes ──
        ctx.strokeStyle = 'rgba(196,154,108,0.70)';
        ctx.lineWidth = 0.9;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, W, H);

        // ── Eyebrows ──
        ctx.strokeStyle = 'rgba(196,154,108,0.55)';
        ctx.lineWidth = 0.9;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, W, H);

        // ── Lips ──
        ctx.strokeStyle = 'rgba(196,120,128,0.65)';
        ctx.lineWidth = 0.9;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, W, H);

        // ── Irises ──
        ctx.strokeStyle = 'rgba(110,207,142,0.60)';
        ctx.lineWidth = 1;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, W, H);

        // ── Iris centre dots ──
        [468, 473].forEach(idx => {
            if (!lm[idx]) return;
            ctx.beginPath();
            ctx.arc(lm[idx].x * W, lm[idx].y * H, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(110,207,142,0.7)';
            ctx.fill();
        });
    }

    if (S.meshStyle === 'minimal') {
        // ── Minimal: just oval + eyes + lips ──
        ctx.strokeStyle = 'rgba(196,154,108,0.4)';
        ctx.lineWidth = 0.8;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, W, H);
        ctx.strokeStyle = 'rgba(196,154,108,0.6)';
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, W, H);
        ctx.strokeStyle = 'rgba(196,120,128,0.55)';
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, W, H);
        ctx.strokeStyle = 'rgba(110,207,142,0.5)';
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, W, H);
    }

    ctx.restore();
}

// ─── MODAL ───────────────────────────────────────────────────────────────────
function openModal() {
    $('overlay').classList.add('open');
    runDeepAnalysis();
}

function closeModal() { $('overlay').classList.remove('open'); }
function maybeClose(e) { if (e.target === $('overlay')) closeModal(); }

async function runDeepAnalysis() {
    $('modal-body').innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <div class="loading-label">Running on-device classifiers…</div>
    </div>`;

    const b64 = canvas.toDataURL('image/jpeg', 0.88).split(',')[1];

    // ── Run ImageClassifier + ImageEmbedder on-device first (lazy load) ──
    let clsLabels = [], embDims = 0;
    try {
        await ensureClassifierEmbedder();
        const src = S.canvasImg || canvas;

        // ImageClassifier — EfficientNet Lite0, ImageNet 1000 labels
        const clsResult = S.imageClassifier.classify(src);
        if (clsResult.classifications?.[0]?.categories) {
            const cats = clsResult.classifications[0].categories;
            S.lastClassification = cats;
            clsLabels = cats.slice(0, 5).map(c =>
                `${c.categoryName.replace(/_/g, ' ')} (${Math.round(c.score * 100)}%)`
            );
            // Show scene classifier card in sidebar
            updateClassifierCard(cats);
        }

        // ImageEmbedder — MobileNet V3, 1024-dim embedding
        const embResult = S.imageEmbedder.embed(src);
        if (embResult.embeddings?.[0]) {
            S.lastEmbedding = embResult.embeddings[0].floatEmbedding;
            embDims = S.lastEmbedding?.length ?? 0;
        }
    } catch (e) {
        console.warn('Classifier/Embedder error:', e);
    }

    $('modal-body').innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <div class="loading-label">Sending to AI model…</div>
    </div>`;

    const ctx_skin = $('sub-skin').textContent;
    const ctx_face = $('val-face').textContent;
    const ctx_temp = $('val-temp').textContent;
    const ctx_orns = Array.from(document.querySelectorAll('.tag')).map(t => t.textContent).join(', ') || 'none';
    const ctx_blend = S.lastBlendshapes.length
        ? S.lastBlendshapes.filter(b => b.score > 0.15).slice(0, 5)
            .map(b => `${b.categoryName}: ${(b.score * 100).toFixed(0)}%`).join(', ')
        : 'none';
    const ctx_cls = clsLabels.length ? clsLabels.join(', ') : 'none';

    const prompt = `You are a professional beauty advisor and facial analysis expert.

On-device sensor readings (pre-computed locally):
• Skin: ${ctx_skin}
• Face shape: ${ctx_face}
• Lighting: ${ctx_temp}
• Ornaments: ${ctx_orns}
• Active expressions (blendshapes): ${ctx_blend}
• Scene classifier labels (EfficientNet): ${ctx_cls}

Analyze the portrait photo and return ONLY a JSON object (no markdown, no preamble):
{
  "description": "2-3 sentence natural, empowering description",
  "skin_analysis": "Tone, undertone, texture, complexion detail",
  "face_features": "Eye shape, brow arch, lip shape, cheekbones, jawline — specific and celebratory",
  "expression_notes": "What the blendshape data and visible expression suggests about this moment",
  "lighting_notes": "How current lighting affects this face and what would be more flattering",
  "accessories": ["list any visible accessories or ornaments"],
  "makeup_tips": ["tip 1", "tip 2", "tip 3", "tip 4"],
  "ornament_recs": ["jewelry style 1", "style 2", "style 3"],
  "color_palette": "3-4 colors that complement this person",
  "skincare_note": "One brief skincare observation",
  "style_tip": "One memorable personalized styling insight"
}`;

    try {
        const res = await fetch(`${BACKEND_URL}/v1/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_b64: b64, prompt })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        // data = { answer: string, model: string, fallback: bool }
        let analysis;
        try { analysis = JSON.parse(data.answer.replace(/```json|```/g, '').trim()); }
        catch { analysis = { _raw: data.answer }; }
        renderAnalysis(analysis, clsLabels, embDims, data.model, data.fallback);
    } catch (err) {
        $('modal-body').innerHTML = `
      <div class="error-box">
        <div class="error-title">Analysis Unavailable</div>
        <div class="error-msg">${err.message}<br><br>
          Make sure the FastAPI proxy is running: cd backend &amp;&amp; uvicorn main:app --reload
        </div>
      </div>`;
    }
}

// ─── CLASSIFIER SIDEBAR CARD ─────────────────────────────────────────────────
function updateClassifierCard(cats) {
    const card = $('card-cls');
    card.style.display = 'block';
    $('val-cls').textContent = cats[0]
        ? cats[0].categoryName.replace(/_/g, ' ')
        : 'Unknown';
    $('cls-list').innerHTML = cats.slice(0, 5).map(c => {
        const pct = Math.round(c.score * 100);
        return `<div class="cls-row">
      <span class="cls-name">${c.categoryName.replace(/_/g, ' ')}</span>
      <div class="cls-score-bg"><div class="cls-score-fill" style="width:${pct}%"></div></div>
      <span class="cls-pct">${pct}%</span>
    </div>`;
    }).join('');
    card.classList.add('lit');
}

function mkList(items) {
    if (!Array.isArray(items) || !items.length) return '<p style="color:var(--muted);font-size:.78rem">—</p>';
    return `<ul class="rec-list">${items.map(i => `<li>${i}</li>`).join('')}</ul>`;
}

function renderAnalysis(a, clsLabels, embDims) {
    if (a._raw) {
        $('modal-body').innerHTML = `<div class="m-section"><div class="m-text" style="white-space:pre-wrap">${a._raw}</div></div>`;
        return;
    }
    $('modal-body').innerHTML = `
    <div class="m-section">
      <span class="m-label">Overview</span>
      <div class="m-text">${a.description || '—'}</div>
    </div>

    <hr class="m-divider">

    <div class="m-grid-2">
      <div class="m-section">
        <span class="m-label">Skin Analysis</span>
        <div class="m-text" style="font-size:.8rem">${a.skin_analysis || '—'}</div>
      </div>
      <div class="m-section">
        <span class="m-label">Facial Features</span>
        <div class="m-text" style="font-size:.8rem">${a.face_features || '—'}</div>
      </div>
    </div>

    ${a.expression_notes ? `
    <div class="m-section">
      <span class="m-label">Expression Reading</span>
      <div class="m-text" style="font-size:.8rem">${a.expression_notes}</div>
    </div>
    <hr class="m-divider">` : '<hr class="m-divider">'}

    <div class="m-section">
      <span class="m-label">Makeup Recommendations</span>
      ${mkList(a.makeup_tips)}
    </div>

    <hr class="m-divider">

    <div class="m-grid-2">
      <div class="m-section">
        <span class="m-label">Ornament Suggestions</span>
        ${mkList(a.ornament_recs)}
      </div>
      <div class="m-section">
        <span class="m-label">Detected Accessories</span>
        ${mkList(a.accessories)}
      </div>
    </div>

    <hr class="m-divider">

    <div class="m-grid-2">
      <div class="m-section">
        <span class="m-label">Complementary Colors</span>
        <div class="m-text" style="font-size:.8rem">${a.color_palette || '—'}</div>
      </div>
      <div class="m-section">
        <span class="m-label">Lighting Notes</span>
        <div class="m-text" style="font-size:.8rem">${a.lighting_notes || '—'}</div>
      </div>
    </div>

    <hr class="m-divider">

    <div class="m-section">
      <span class="m-label">Skincare</span>
      <div class="m-text" style="font-size:.8rem">${a.skincare_note || '—'}</div>
    </div>

    <div class="m-section">
      <span class="m-label">Style Insight</span>
      <div class="style-insight">${a.style_tip || '—'}</div>
    </div>

    ${clsLabels.length || embDims ? `
    <hr class="m-divider">
    <div class="m-grid-2">
      ${clsLabels.length ? `
      <div class="m-section">
        <span class="m-label">Scene Classifier · EfficientNet Lite0</span>
        <div class="m-text" style="font-size:.75rem;line-height:1.8">${clsLabels.join('<br>')}</div>
      </div>` : ''}
      ${embDims ? `
      <div class="m-section">
        <span class="m-label">Image Embedding · MobileNet V3</span>
        <div class="m-text" style="font-size:.75rem">${embDims}-dim vector extracted.<br>
          <span style="color:var(--muted)">Can be used for face similarity matching in future versions.</span>
        </div>
      </div>` : ''}
    </div>` : ''}
  `;
}

// ─── BOOT ────────────────────────────────────────────────────────────────────
setStatus('processing', 'Loading MediaPipe Tasks Vision…');
initAllTasks();
