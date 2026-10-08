# -*- coding: utf-8 -*-
"""
BilinGuard Web Application v5.0
Flask backend with REST API for video/image upload, full jaundice
assessment pipeline, clinical decision support and interpretability.
"""
import os, sys, time, json, uuid, base64
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PHI-safe CDSS audit trail (closed-loop logging; see clinical_advisor.py v3)
AUDIT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'audit', 'cdss_advice_log.jsonl')
try:
    os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
except Exception:
    pass

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from inference import BilinGuardPredictor, encode_image_base64, analyze_color

try:
    import cv2
    import numpy as np
    HAS_CV = True
except Exception:
    HAS_CV = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from clinical_advisor import ClinicalAdvisor, DEPARTMENTS
    HAS_ADVISOR = True
except Exception:
    ClinicalAdvisor = None
    DEPARTMENTS = {}
    HAS_ADVISOR = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB max upload

UPLOAD_DIR = os.path.join(os.environ.get('TEMP', '/tmp'), 'bilinguard_uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(UPLOAD_DIR, exist_ok=True)

print('Initializing BilinGuard inference engine...')
predictor = BilinGuardPredictor(model_dir=MODEL_DIR if os.path.exists(MODEL_DIR) else None)
print(f'Model mode: {"trained" if predictor.is_trained else "demo"}')

ALLOWED_VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_IMAGE = {'.jpg', '.jpeg', '.png', '.bmp'}

# Pending interpretability sessions: token -> {face, eyelid, frame, ts}
SESSIONS = {}
SESSION_TTL = 900  # seconds


def _reap_sessions():
    now = time.time()
    for tok, sess in list(SESSIONS.items()):
        if now - sess.get('ts', 0) > SESSION_TTL:
            for key in ('face', 'eyelid', 'frame'):
                _safe_remove(sess.get(key))
            SESSIONS.pop(tok, None)


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
def _save_upload(file_storage, suffix=''):
    ext = os.path.splitext(file_storage.filename)[1].lower()
    filename = f'{uuid.uuid4().hex}{suffix}{ext}'
    filepath = os.path.join(UPLOAD_DIR, filename)
    file_storage.save(filepath)
    return filepath, ext


def _safe_remove(path):
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def _read_image_rgb(path):
    fb = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _build_tasks(face_path, eyelid_path, is_video):
    """Ordered (key, callable) model-group tasks, fastest first."""
    p = predictor

    def run_task(video_fn, image_fn, eyelid_fn=None):
        def _run():
            try:
                if is_video and face_path:
                    return video_fn()
                elif eyelid_path and eyelid_fn:
                    return eyelid_fn()
                elif face_path and image_fn:
                    return image_fn()
                else:
                    return {'error': 'No input'}
            except Exception as e:
                return {'error': str(e)}
        return _run

    return [
        ('screen', run_task(lambda: p.analyze_video(face_path, 'screen'),
                            lambda: p.screen_face(face_path))),
        ('grade', run_task(lambda: p.analyze_video(face_path, 'grade'),
                           lambda: p.classify_grade_face(face_path),
                           lambda: p.analyze_eyelid(eyelid_path))),
        ('dbil', run_task(lambda: p.analyze_video(face_path, 'dbil'),
                          lambda: p.classify_dbil_face(face_path),
                          lambda: p.classify_dbil_eyelid(eyelid_path))),
        ('ibil', run_task(lambda: p.analyze_video(face_path, 'ibil'),
                          lambda: p.classify_ibil_face(face_path),
                          lambda: p.classify_ibil_eyelid(eyelid_path))),
        ('type', run_task(lambda: p.analyze_video(face_path, 'type'),
                          lambda: p.classify_type(face_path))),
        ('cp', run_task(lambda: p.analyze_video(face_path, 'cp'),
                        lambda: p.classify_child_pugh(face_path))),
        ('meld', run_task(lambda: p.analyze_video(face_path, 'meld'),
                          lambda: p.classify_meld(face_path))),
    ]


def _build_eyelid_tasks(eyelid_path):
    """Eyelid-only pipeline: every task runs on the everted-eyelid photo."""
    p = predictor
    return [
        ('screen', lambda: p.screen_eyelid(eyelid_path)),
        ('grade', lambda: p.analyze_eyelid(eyelid_path)),
        ('dbil', lambda: p.classify_dbil_eyelid(eyelid_path)),
        ('ibil', lambda: p.classify_ibil_eyelid(eyelid_path)),
        ('type', lambda: p.classify_type_eyelid(eyelid_path)),
        ('cp', lambda: p.classify_child_pugh_eyelid(eyelid_path)),
        ('meld', lambda: p.classify_meld_eyelid(eyelid_path)),
    ]


def _extract_mid_frame(video_path):
    """Pull the middle frame out of a video for Grad-CAM."""
    try:
        frames = predictor.extract_video_frames(video_path, max_frames=12)
        if frames:
            import tempfile
            mid = len(frames) // 2
            frame_path = os.path.join(tempfile.gettempdir(), f'bilinguard_frame_{uuid.uuid4().hex}.jpg')
            cv2.imwrite(frame_path, cv2.cvtColor(frames[mid], cv2.COLOR_RGB2BGR))
            return frame_path
    except Exception:
        pass
    return None


def _compute_gradcam(image_path, source_type='face'):
    """Grad-CAM original / overlay / heatmap as base64 (desktop parity)."""
    if not (HAS_CV and HAS_TORCH) or not image_path or not os.path.exists(image_path):
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.cm as cm
        from inference import GradCAMGenerator

        img_rgb = _read_image_rgb(image_path)
        if img_rgb is None:
            return None

        use_sam = source_type == 'face' and getattr(predictor, 'sam_model', None) is not None
        if use_sam:
            processed = predictor._preprocess_face_sam(img_rgb)
        else:
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
            processed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
            processed = cv2.resize(processed, (224, 224))

        disp = processed.copy()
        cam_model = None
        if source_type == 'face':
            for name, m in getattr(predictor, 'face_binary_models', []):
                if 'convnext' in name.lower():
                    cam_model = m
                    break
        if cam_model is None:
            for i in range(min(len(predictor.model_names), len(predictor.models))):
                if 'convnext' in predictor.model_names[i].lower():
                    cam_model = predictor.models[i]
                    break

        overlay = heatmap = None
        if cam_model is not None:
            gen = GradCAMGenerator(cam_model)
            norm = processed.astype(np.float32) / 255.0
            norm = (norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            t = torch.FloatTensor(np.stack([norm.transpose(2, 0, 1)])).to(predictor.device)
            cam = gen.generate(t)
            if cam is not None:
                hm = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
                overlay = (disp * 0.5 + hm * 0.5).astype(np.uint8)
                heatmap = hm

        return {
            'original': encode_image_base64(disp),
            'overlay': encode_image_base64(overlay) if overlay is not None else None,
            'heatmap': encode_image_base64(heatmap) if heatmap is not None else None,
        }
    except Exception as e:
        print('[Grad-CAM]', e, flush=True)
        return None


def _compute_agreement(gradcam_img, models=None):
    """Per-binary-model votes + entropy / margin / agreement.
    `models` defaults to face binary models; pass eyelid_binary_models for
    eyelid-only mode. `gradcam_img` is a grad-cam dict with an 'original' key."""
    if not (HAS_TORCH and gradcam_img):
        return None, None
    if models is None:
        models = getattr(predictor, 'face_binary_models', None)
    if not models:
        return None, None
    try:
        fb = base64.b64decode(gradcam_img['original'])
        arr = np.frombuffer(fb, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        norm = processed.astype(np.float32) / 255.0
        norm = (norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        t = torch.FloatTensor(np.stack([norm.transpose(2, 0, 1)])).to(predictor.device)

        individual = []
        with torch.no_grad():
            for name, m in models:
                out = m(t)
                pr = F.softmax(out, dim=1).mean(dim=0).cpu().numpy()
                pj = float(pr[1]) if len(pr) > 1 else float(pr[0])
                individual.append([name, 1 if pj > 0.5 else 0, pj])
        if not individual:
            return None, None
        ens = float(np.mean([x[2] for x in individual]))
        p = min(max(ens, 1e-6), 1 - 1e-6)
        entropy = float(-(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2))
        margin = abs(ens - 0.5) * 2
        preds = [x[1] for x in individual]
        agreement = max(preds.count(0), preds.count(1)) / len(preds)
        return individual, {'entropy': entropy, 'margin': float(margin), 'agreement': float(agreement)}
    except Exception as e:
        print('[agreement]', e, flush=True)
        return None, None


# ════════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


@app.after_request
def add_cache_headers(resp):
    if request.path.startswith('/static/'):
        resp.headers['Cache-Control'] = 'no-cache'
    return resp


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_mode': 'trained' if predictor.is_trained else 'demo',
        'gpu_available': predictor.device is not None and str(predictor.device) == 'cuda',
        'n_models': len(predictor.models),
        'n_face_binary_models': len(predictor.face_binary_models),
        'n_type_class_models': len(predictor.type_class_models),
        'has_advisor': HAS_ADVISOR,
    })


@app.route('/api/departments')
def departments():
    items = []
    for dept_id, info in DEPARTMENTS.items():
        items.append({
            'id': dept_id,
            'name_en': info.get('name_en', dept_id),
            'name_cn': info.get('name_cn', dept_id),
        })
    return jsonify({'departments': items})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Streaming full pipeline: NDJSON, one line per model group."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO and ext not in ALLOWED_IMAGE:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    face_path, _ = _save_upload(file)
    is_video = ext in ALLOWED_VIDEO

    eyelid_path = None
    if 'eyelid' in request.files and request.files['eyelid'].filename:
        eye_file = request.files['eyelid']
        eye_ext = os.path.splitext(eye_file.filename)[1].lower()
        if eye_ext in ALLOWED_IMAGE:
            eyelid_path, _ = _save_upload(eye_file, suffix='_eyelid')

    lang = request.form.get('lang', 'cn')
    if lang not in ('cn', 'en'):
        lang = 'cn'
    department = request.form.get('department') or None
    diagnosis = request.form.get('diagnosis') or None
    try:
        lab_values = json.loads(request.form.get('labs', '{}') or '{}')
        lab_values = {k: float(v) for k, v in lab_values.items() if v not in (None, '')}
    except Exception:
        lab_values = {}
    try:
        _nut_raw = (request.form.get('nutrition') or '').strip()
        nutrition = json.loads(_nut_raw) if _nut_raw else None
    except Exception:
        nutrition = None

    def generate():
        t0 = time.time()
        results = {}
        try:
            _reap_sessions()
            for key, fn in _build_tasks(face_path, eyelid_path, is_video):
                results[key] = fn()
                yield json.dumps({'type': 'group', 'key': key, 'data': results[key]},
                                 ensure_ascii=False) + '\n'

            advice = None
            if HAS_ADVISOR:
                try:
                    advisor = ClinicalAdvisor(lang=lang, department=department,
                                              audit_log=AUDIT_LOG)
                    advice = advisor.advise(results, lab_values=lab_values,
                                            department=department, diagnosis=diagnosis,
                                            nutrition=nutrition)
                except Exception as e:
                    print('[advisor]', e, flush=True)
            yield json.dumps({'type': 'advice', 'data': advice}, ensure_ascii=False) + '\n'

            token = uuid.uuid4().hex
            SESSIONS[token] = {
                'face': face_path,
                'eyelid': eyelid_path,
                'frame': None,
                'is_video': is_video,
                'ts': time.time(),
            }
            yield json.dumps({
                'type': 'done',
                'token': token,
                'inference_time': round(time.time() - t0, 2),
                'input_mode': 'video' if is_video else ('combined' if eyelid_path else 'face'),
            }, ensure_ascii=False) + '\n'
        except Exception as e:
            _safe_remove(face_path)
            _safe_remove(eyelid_path)
            yield json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False) + '\n'

    from flask import Response
    return Response(generate(), mimetype='application/x-ndjson',
                    headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})


@app.route('/api/analyze_eyelid', methods=['POST'])
def analyze_eyelid():
    """Eyelid-only streaming pipeline (NDJSON). Runs all 7 tasks on the
    everted-eyelid photo — no face image required."""
    if 'eyelid' not in request.files or not request.files['eyelid'].filename:
        return jsonify({'error': 'No eyelid image uploaded'}), 400
    eye_file = request.files['eyelid']
    eye_ext = os.path.splitext(eye_file.filename)[1].lower()
    if eye_ext not in ALLOWED_IMAGE:
        return jsonify({'error': f'Image format required: {eye_ext}'}), 400
    eyelid_path, _ = _save_upload(eye_file, suffix='_eyelid')

    lang = request.form.get('lang', 'cn')
    if lang not in ('cn', 'en'):
        lang = 'cn'
    department = request.form.get('department') or None
    diagnosis = request.form.get('diagnosis') or None
    try:
        lab_values = json.loads(request.form.get('labs', '{}') or '{}')
        lab_values = {k: float(v) for k, v in lab_values.items() if v not in (None, '')}
    except Exception:
        lab_values = {}
    try:
        _nut_raw = (request.form.get('nutrition') or '').strip()
        nutrition = json.loads(_nut_raw) if _nut_raw else None
    except Exception:
        nutrition = None

    def generate():
        t0 = time.time()
        results = {}
        try:
            _reap_sessions()
            for key, fn in _build_eyelid_tasks(eyelid_path):
                try:
                    results[key] = fn()
                except Exception as e:
                    results[key] = {'error': str(e)}
                yield json.dumps({'type': 'group', 'key': key, 'data': results[key]},
                                 ensure_ascii=False) + '\n'

            advice = None
            if HAS_ADVISOR:
                try:
                    advisor = ClinicalAdvisor(lang=lang, department=department,
                                              audit_log=AUDIT_LOG)
                    advice = advisor.advise(results, lab_values=lab_values,
                                            department=department, diagnosis=diagnosis,
                                            nutrition=nutrition)
                except Exception as e:
                    print('[advisor]', e, flush=True)
            yield json.dumps({'type': 'advice', 'data': advice}, ensure_ascii=False) + '\n'

            token = uuid.uuid4().hex
            SESSIONS[token] = {
                'face': None,
                'eyelid': eyelid_path,
                'frame': None,
                'is_video': False,
                'mode': 'eyelid',
                'ts': time.time(),
            }
            yield json.dumps({
                'type': 'done',
                'token': token,
                'inference_time': round(time.time() - t0, 2),
                'input_mode': 'eyelid',
            }, ensure_ascii=False) + '\n'
        except Exception as e:
            _safe_remove(eyelid_path)
            yield json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False) + '\n'

    from flask import Response
    return Response(generate(), mimetype='application/x-ndjson',
                    headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})


@app.route('/api/interpret', methods=['POST'])
def interpret():
    """Second phase: Grad-CAM + colour + agreement for a finished analysis."""
    data = request.get_json(silent=True) or {}
    token = data.get('token') or request.form.get('token')
    sess = SESSIONS.pop(token, None)
    if not sess:
        return jsonify({'error': 'Session expired or invalid'}), 410

    face_path = sess.get('face')
    eyelid_path = sess.get('eyelid')
    frame_path = sess.get('frame')
    mode = sess.get('mode', 'face')
    if not frame_path and sess.get('is_video') and face_path:
        frame_path = _extract_mid_frame(face_path)
    if not frame_path:
        frame_path = face_path
    try:
        interp = {}
        face_gc = None
        if mode != 'eyelid' and frame_path:
            face_gc = _compute_gradcam(frame_path, 'face')
            if face_gc:
                interp['face'] = face_gc
        eye_gc = None
        if eyelid_path:
            eye_gc = _compute_gradcam(eyelid_path, 'eyelid')
            if eye_gc:
                interp['eyelid'] = eye_gc

        color_src = eyelid_path or frame_path
        if HAS_CV and color_src and os.path.exists(color_src):
            img_rgb = _read_image_rgb(color_src)
            if img_rgb is not None:
                interp['color'] = analyze_color(img_rgb)

        # Agreement: eyelid binary models in eyelid mode, face binary otherwise.
        if mode == 'eyelid':
            agree_on, agree_models = eye_gc, getattr(predictor, 'eyelid_binary_models', [])
        else:
            agree_on, agree_models = face_gc, None
        individual, confidence = _compute_agreement(agree_on, agree_models)
        if individual:
            interp['individual'] = individual
        if confidence:
            interp['confidence'] = confidence

        return jsonify({'interp': interp})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _safe_remove(face_path)
        _safe_remove(eyelid_path)
        if frame_path and frame_path != face_path:
            _safe_remove(frame_path)


# ── Legacy endpoints (kept for backward compatibility) ───────────────
@app.route('/api/screen', methods=['POST'])
def screen():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE:
        return jsonify({'error': f'Image format required: {ext}'}), 400
    filepath, _ = _save_upload(file)
    try:
        result = predictor.screen_face(filepath)
        if 'error' in result:
            return jsonify(result), 422
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _safe_remove(filepath)


@app.route('/api/type', methods=['POST'])
def classify_type():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE:
        return jsonify({'error': f'Image format required: {ext}'}), 400
    filepath, _ = _save_upload(file)
    try:
        result = predictor.classify_type(filepath)
        if 'error' in result:
            return jsonify(result), 422
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _safe_remove(filepath)


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO and ext not in ALLOWED_IMAGE:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    filepath, _ = _save_upload(file)
    is_video = ext in ALLOWED_VIDEO

    eyelid_path = None
    if 'eyelid' in request.files and request.files['eyelid'].filename:
        eye_file = request.files['eyelid']
        eye_ext = os.path.splitext(eye_file.filename)[1].lower()
        if eye_ext in ALLOWED_IMAGE:
            eyelid_path, _ = _save_upload(eye_file, suffix='_eyelid')

    mode = request.form.get('mode', 'auto')
    try:
        if mode == 'eyelid' or (eyelid_path and not filepath):
            result = predictor.analyze_eyelid(eyelid_path if eyelid_path else filepath)
        elif eyelid_path:
            result = predictor.analyze_combined(filepath, eyelid_path, is_video=is_video)
        else:
            result = predictor.analyze_face(filepath, is_video=is_video)
        if 'error' in result:
            return jsonify(result), 422
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _safe_remove(filepath)
        _safe_remove(eyelid_path)


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    files = request.files.getlist('files')
    results = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        is_video = ext in ALLOWED_VIDEO
        filepath, _ = _save_upload(file)
        try:
            result = predictor.analyze(filepath, is_video=is_video)
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})
        finally:
            _safe_remove(filepath)
    return jsonify({'results': results})


if __name__ == '__main__':
    import socket
    port = int(os.environ.get('PORT', 5000))
    print('\n' + '=' * 50)
    print('  BilinGuard AI Jaundice Assessment System')
    print('=' * 50)
    print(f'  Local access:  http://127.0.0.1:{port}')
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, port, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith('127.') and not ip.startswith('169.254'):
                print(f'  LAN access:    http://{ip}:{port}')
    except Exception:
        pass
    print('=' * 50)
    print('  Other computers on the same network can use the LAN URL.')
    print('  If access fails, run setup_firewall.bat as Administrator.')
    print('=' * 50 + '\n')
    app.run(host='0.0.0.0', port=port, debug=False)
