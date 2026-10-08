# -*- coding: utf-8 -*-
"""
Audit Tool: Compare v3 (broken) vs v4 (fixed) face preprocessing.

Produces:
  1. Detection rate stats per category
  2. Skin fraction comparison (v3 vs v4) 
  3. Sample overlay images: raw frame + detection bbox + landmarks
  4. Side-by-side comparison grid: raw | v3 crop | v4 crop
  5. Summary report
"""

import os, sys, json, random, argparse
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'D:\research\人脸识别营养\传染科'
VF_DIR = os.path.join(BASE, 'data', 'video_frames')
V3_DIR = os.path.join(BASE, 'data', 'sam_processed_v3')
V4_DIR = os.path.join(BASE, 'data', 'face_v4')
MODEL_DIR = os.path.join(BASE, 'models')
RESULT_DIR = os.path.join(BASE, 'results', 'audit_v4')

TEMP_DIR = r'C:\Users\o\AppData\Local\Temp\opencode'
YUNET_TMP = os.path.join(TEMP_DIR, 'yunet.onnx')

SEED = 42
random.seed(SEED); np.random.seed(SEED)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 9,
    'axes.linewidth': 0.8, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def skin_fraction(img_rgb):
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    cb = ycbcr[:, :, 2].astype(int)
    cr = ycbcr[:, :, 1].astype(int)
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return skin.mean()


def read_img(path):
    fb = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_v3_crops(cat, pid):
    """Get existing v3 face crops for a patient."""
    d = os.path.join(V3_DIR, cat, pid)
    if not os.path.isdir(d): return []
    return sorted([os.path.join(d, f) for f in os.listdir(d) if '_face.jpg' in f])


def get_v4_crops(cat, pid):
    d = os.path.join(V4_DIR, cat, pid)
    if not os.path.isdir(d): return []
    paths = sorted([os.path.join(d, f) for f in os.listdir(d) if '_face.jpg' in f])
    metas = []
    for p in paths:
        mp = p.replace('_face.jpg', '_face_meta.json')
        if os.path.exists(mp):
            with open(mp, 'r') as mf:
                metas.append(json.load(mf))
        else:
            metas.append(None)
    return list(zip(paths, metas))


def test_detection(detector, cat, pid):
    """Run YuNet detection on all frames of a patient, count results."""
    vf_p = os.path.join(VF_DIR, cat, pid)
    if not os.path.isdir(vf_p): return None
    frames = sorted([f for f in os.listdir(vf_p) if f.endswith('.jpg')])
    results = {'total': len(frames), 'detected': 0, 'failed': 0,
               'det_scores': [], 'bbox_sizes': []}
    for fname in frames:
        img = read_img(os.path.join(vf_p, fname))
        if img is None: continue
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(bgr)
        if faces is not None and len(faces) > 0:
            results['detected'] += 1
            best = faces[np.argmax(faces[:, -1])]
            results['det_scores'].append(float(best[-1]))
            fw, fh = best[2], best[3]
            results['bbox_sizes'].append((fw * fh) / (w * h))
        else:
            results['failed'] += 1
    return results


def main():
    parser = argparse.ArgumentParser(description="Audit v4 face preprocessing")
    parser.add_argument('--sample', type=int, default=30,
                        help="Patients to audit per category")
    parser.add_argument('--viz', type=int, default=8,
                        help="Visualization samples per category")
    parser.add_argument('--no-viz', action='store_true',
                        help="Skip visualization generation")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ── Setup detector ──
    if not os.path.exists(YUNET_TMP):
        print(f"[FATAL] YuNet model not found at {YUNET_TMP}")
        print("Run face_preprocess_v4.py first or copy model manually.")
        return

    detector = cv2.FaceDetectorYN.create(
        YUNET_TMP, "", (640, 480),
        score_threshold=0.50, nms_threshold=0.3, top_k=5000)
    print("[Setup] YuNet ready for audit")

    # ── Gather patients ──
    print("\n[1] Gathering patients...")
    patients = {}
    for cat in ['normal', 'mild', 'moderate', 'severe']:
        vf_cat = os.path.join(VF_DIR, cat)
        v4_cat = os.path.join(V4_DIR, cat)
        if not os.path.isdir(vf_cat): continue
        pids = sorted([d for d in os.listdir(vf_cat)
                       if os.path.isdir(os.path.join(vf_cat, d))])
        random.shuffle(pids)
        pids = pids[:args.sample]
        patients[cat] = []
        for pid in pids:
            has_v4 = os.path.isdir(os.path.join(v4_cat, pid))
            patients[cat].append({'pid': pid, 'has_v4': has_v4})
        print(f"  {cat}: {len(patients[cat])} patients "
              f"(v4 processed: {sum(1 for p in patients[cat] if p['has_v4'])})")

    # ── Detection rate audit ──
    print("\n[2] Auditing detection rates...")
    det_report = []
    for cat, pats in patients.items():
        for p in pats:
            r = test_detection(detector, cat, p['pid'])
            if r is None: continue
            det_rate = r['detected'] / max(r['total'], 1)
            avg_score = np.mean(r['det_scores']) if r['det_scores'] else 0
            avg_bbox_frac = np.mean(r['bbox_sizes']) if r['bbox_sizes'] else 0
            det_report.append({
                'category': cat, 'patient_id': p['pid'],
                'has_v4': p['has_v4'],
                'frames': r['total'],
                'faces_detected': r['detected'],
                'detection_rate': round(det_rate, 3),
                'avg_score': round(avg_score, 3),
                'avg_bbox_frac': round(avg_bbox_frac, 4),
            })

    det_df = pd.DataFrame(det_report)
    det_df.to_csv(os.path.join(RESULT_DIR, 'detection_rates.csv'),
                  index=False, encoding='utf-8-sig')

    print("\n  Detection rate summary:")
    for cat in det_df['category'].unique():
        sub = det_df[det_df['category'] == cat]
        print(f"    {cat:12s}: {len(sub)} patients, "
              f"det_rate={sub['detection_rate'].mean():.1%}, "
              f"avg_score={sub['avg_score'].mean():.3f}, "
              f"avg_bbox_area={sub['avg_bbox_frac'].mean()*100:.1f}% of frame")

    # ── Skin fraction comparison ──
    print("\n[3] Comparing skin fractions (v3 vs v4)...")
    skin_report = []
    for cat, pats in patients.items():
        for p in pats:
            v3_crops = get_v3_crops(cat, p['pid'])
            v4_crops = get_v4_crops(cat, p['pid'])
            v3_skins = []
            for cp in v3_crops[:10]:
                img = read_img(cp)
                if img is not None:
                    v3_skins.append(skin_fraction(img))
            v4_skins = []
            for cp, _ in v4_crops[:10]:
                img = read_img(cp)
                if img is not None:
                    v4_skins.append(skin_fraction(img))
            skin_report.append({
                'category': cat, 'patient_id': p['pid'],
                'v3_n': len(v3_skins),
                'v3_skin_mean': round(np.mean(v3_skins), 3) if v3_skins else None,
                'v3_skin_std': round(np.std(v3_skins), 3) if v3_skins else None,
                'v4_n': len(v4_skins),
                'v4_skin_mean': round(np.mean(v4_skins), 3) if v4_skins else None,
                'v4_skin_std': round(np.std(v4_skins), 3) if v4_skins else None,
            })

    skin_df = pd.DataFrame(skin_report)
    skin_df.to_csv(os.path.join(RESULT_DIR, 'skin_fraction_comparison.csv'),
                   index=False, encoding='utf-8-sig')

    print("\n  Skin fraction summary (mean of patient means):")
    for cat in skin_df['category'].unique():
        sub = skin_df[skin_df['category'] == cat]
        v3_valid = sub[sub['v3_skin_mean'].notna()]
        v4_valid = sub[sub['v4_skin_mean'].notna()]
        v3_m = v3_valid['v3_skin_mean'].mean() if len(v3_valid) > 0 else 0
        v4_m = v4_valid['v4_skin_mean'].mean() if len(v4_valid) > 0 else 0
        print(f"    {cat:12s}: v3 skin={v3_m:.1%}  v4 skin={v4_m:.1%}  Δ={v4_m-v3_m:+.1%}")

    # Check if v4 data exists for visualization
    any_v4 = any(p['has_v4'] for cat_pats in patients.values() for p in cat_pats)
    if not any_v4:
        print("\n[4] No v4 data available for visualization. "
              "Run face_preprocess_v4.py first.")
        if args.no_viz:
            return

    # ── Skin fraction bar chart ──
    print("\n[4] Generating skin fraction comparison chart...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    cats = ['normal', 'mild', 'moderate', 'severe']
    v3_means = []; v4_means = []
    for cat in cats:
        sub = skin_df[skin_df['category'] == cat]
        v3_v = sub[sub['v3_skin_mean'].notna()]['v3_skin_mean']
        v4_v = sub[sub['v4_skin_mean'].notna()]['v4_skin_mean']
        v3_means.append(v3_v.mean() if len(v3_v) > 0 else 0)
        v4_means.append(v4_v.mean() if len(v4_v) > 0 else 0)

    x = np.arange(len(cats))
    w = 0.35
    axes[0].bar(x - w/2, [m * 100 for m in v3_means], w, label='v3 (broken)', color='#E74C3C', alpha=0.8)
    axes[0].bar(x + w/2, [m * 100 for m in v4_means], w, label='v4 (fixed)', color='#2ECC71', alpha=0.8)
    axes[0].axhline(y=35, color='gray', linestyle='--', alpha=0.5, label='min threshold (35%)')
    axes[0].set_ylabel('Skin Fraction (%)')
    axes[0].set_title('Skin Fraction: v3 vs v4', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, max(100, max(v3_means+v4_means)*120))

    # Distribution for one category (e.g., combined)
    all_v3 = skin_df[skin_df['v3_skin_mean'].notna()]['v3_skin_mean']
    all_v4 = skin_df[skin_df['v4_skin_mean'].notna()]['v4_skin_mean']
    axes[1].hist(all_v3, bins=20, alpha=0.5, color='#E74C3C', label=f'v3 (n={len(all_v3)})',
                 edgecolor='white', linewidth=0.5)
    if len(all_v4) > 0:
        axes[1].hist(all_v4, bins=20, alpha=0.5, color='#2ECC71', label=f'v4 (n={len(all_v4)})',
                     edgecolor='white', linewidth=0.5)
    axes[1].axvline(x=0.35, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Skin Fraction')
    axes[1].set_ylabel('Patient Count')
    axes[1].set_title('Skin Fraction Distribution (per-patient mean)', fontweight='bold')
    axes[1].legend(fontsize=8)

    fig.suptitle('Face Crop Quality: v3 (Broken Haarcascade) vs v4 (YuNet + SAM)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'skin_fraction_comparison.png'))
    fig.savefig(os.path.join(RESULT_DIR, 'skin_fraction_comparison.svg'))
    plt.close(fig)
    print("  Saved skin fraction chart")

    # ── Detection overlay samples ──
    if args.no_viz:
        print("\n[5] Visualization skipped (--no-viz)")
    else:
        print("\n[5] Generating detection overlay samples...")
        for cat in cats:
            if cat not in patients: continue
            pats = [p for p in patients[cat] if p['has_v4']]
            if not pats: continue
            sample_pats = random.sample(pats, min(args.viz, len(pats)))

            n_cols = min(4, len(sample_pats))
            n_rows = len(sample_pats)
            fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for row, p in enumerate(sample_pats):
                pid = p['pid']
                # Find a frame where detection worked
                vf_p = os.path.join(VF_DIR, cat, pid)
                v4_p = os.path.join(V4_DIR, cat, pid)
                if not os.path.isdir(vf_p): continue

                # Pick a frame that was successfully processed in v4
                v4_faces = sorted([f for f in os.listdir(v4_p) if '_face.jpg' in f])
                if not v4_faces: continue
                chosen_v4 = random.choice(v4_faces)
                frame_name = chosen_v4.replace('_face.jpg', '.jpg')
                frame_path = os.path.join(vf_p, frame_name)

                if not os.path.exists(frame_path):
                    # Try any frame
                    frames = sorted([f for f in os.listdir(vf_p) if f.endswith('.jpg')])
                    if not frames: continue
                    frame_path = os.path.join(vf_p, random.choice(frames))
                    frame_name = os.path.basename(frame_path)

                # 1. Raw frame with detection overlay
                raw = read_img(frame_path)
                if raw is None: continue
                bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                h, w = raw.shape[:2]
                detector.setInputSize((w, h))
                _, faces = detector.detect(bgr)
                overlay = raw.copy()
                if faces is not None:
                    # Draw all detected faces
                    for fc in faces:
                        bx, by, bw_, bh_ = fc[:4].astype(int)
                        score = fc[-1]
                        lm = fc[4:14].reshape(5, 2)
                        color = (46, 204, 113) if score >= 0.5 else (231, 76, 60)
                        cv2.rectangle(overlay, (bx, by), (bx + bw_, by + bh_), color, 3)
                        cv2.putText(overlay, f'{score:.2f}', (bx, by - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        for (lx, ly) in lm:
                            cv2.circle(overlay, (int(lx), int(ly)), 3, (52, 152, 219), -1)

                axes[row, 0].imshow(overlay)
                axes[row, 0].set_title(f'Detection ({len(faces) if faces is not None else 0} faces)', fontsize=8, fontweight='bold')
                axes[row, 0].axis('off')

                # 2. v3 crop
                v3_path = os.path.join(V3_DIR, cat, pid, frame_name.replace('.jpg', '_face.jpg'))
                if os.path.exists(v3_path):
                    v3_img = read_img(v3_path)
                    if v3_img is not None:
                        axes[row, 1].imshow(v3_img)
                        sf3 = skin_fraction(v3_img)
                        axes[row, 1].set_title(f'v3 crop (skin={sf3:.0%})', fontsize=8)
                    else:
                        axes[row, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
                else:
                    axes[row, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[row, 1].axis('off')

                # 3. v4 crop
                v4_face_path = os.path.join(V4_DIR, cat, pid,
                                             frame_name.replace('.jpg', '_face.jpg'))
                if os.path.exists(v4_face_path):
                    v4_img = read_img(v4_face_path)
                    if v4_img is not None:
                        axes[row, 2].imshow(v4_img)
                        sf4 = skin_fraction(v4_img)
                        axes[row, 2].set_title(f'v4 crop (skin={sf4:.0%})', fontsize=8)
                    else:
                        axes[row, 2].text(0.5, 0.5, 'N/A', ha='center', va='center')
                else:
                    axes[row, 2].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[row, 2].axis('off')

                # 4. v4 crop with CLAHE
                axes[row, 3].imshow(v4_img if v4_img is not None else np.zeros((100,100,3)))
                axes[row, 3].set_title('v4 (CLAHE)', fontsize=8)
                axes[row, 3].axis('off')

                # Row label
                axes[row, 0].set_ylabel(f'{pid[:20]}\n{cat}', fontsize=7, rotation=0,
                                       labelpad=80, va='center')

            col_titles = ['YuNet Detection', 'v3 "Face" Crop', 'v4 Face Crop', 'v4 CLAHE']
            for col, title in enumerate(col_titles):
                axes[0, col].set_title(title, fontsize=9, fontweight='bold', color='#2C3E50')

            fig.suptitle(f'Sample Comparison: v3 vs v4 — {cat} patients',
                         fontsize=11, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0.1, 0, 1, 0.97])
            out_p = os.path.join(RESULT_DIR, f'comparison_{cat}.png')
            fig.savefig(out_p)
            fig.savefig(out_p.replace('.png', '.svg'))
            plt.close(fig)
            print(f"  Saved: comparison_{cat}.png")

    # ── Summary report ──
    print("\n" + "=" * 60)
    print("  Audit Complete")
    print("=" * 60)
    print(f"  Results saved to: {RESULT_DIR}")
    for f in sorted(os.listdir(RESULT_DIR)):
        p = os.path.join(RESULT_DIR, f)
        if os.path.isfile(p):
            print(f"    {f} ({os.path.getsize(p)//1024} KB)")


if __name__ == '__main__':
    main()
