#!/usr/bin/env python
"""Sequential-process FP32 Paddle benchmark; outputs are not accuracy ground truth."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'v5_mobile_latin': ('PP-OCRv5_mobile_det', 'latin_PP-OCRv5_mobile_rec'),
    'v6_small': ('PP-OCRv6_small_det', 'PP-OCRv6_small_rec'),
    'v6_medium': ('PP-OCRv6_medium_det', 'PP-OCRv6_medium_rec'),
}
SETTINGS = dict(device='gpu:0', engine='paddle_static',
                engine_config={'run_mode': 'paddle'},
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=True)


def worker(args):
    import cv2
    import numpy as np
    import paddle
    import paddleocr
    sys.path.insert(0, str(ROOT))
    from ocr_annotate import NUMERIC_TEXT, annotate, load_font
    from PIL import Image
    manifest = json.loads((args.output / 'frames.json').read_text())
    frames = [cv2.imread(str(args.output / f['file'])) for f in manifest['frames']]
    if any(f is None for f in frames):
        raise RuntimeError('Frame loading failed')
    det, rec = MODELS[args.worker]
    # Official Paddle built-in downloader (no ONNX); names explicitly selected.
    ocr = paddleocr.PaddleOCR(text_detection_model_name=det,
                             text_recognition_model_name=rec, **SETTINGS)
    for frame in frames:
        list(ocr.predict(frame))
        paddle.device.synchronize()
    records = []
    for round_index in range(args.rounds):
        for index, frame in enumerate(frames):
            paddle.device.synchronize()
            start = perf_counter()
            result = list(ocr.predict(frame))[0]
            paddle.device.synchronize()
            elapsed = (perf_counter() - start) * 1000
            data = result.json['res']
            texts = data['rec_texts']
            records.append(dict(round=round_index, frame=index, ms=elapsed,
                                detection_count=len(data['dt_polys']),
                                recognition_count=len(texts), texts=texts,
                                numeric_texts=[t for t in texts if t and NUMERIC_TEXT.fullmatch(t.strip())],
                                effective_config={key: data[key] for key in
                                                  ('model_settings', 'text_det_params', 'text_rec_score_thresh')
                                                  if key in data}))
            if round_index == 0:
                image = annotate(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                 data, load_font(30))
                image.save(args.output / f'{args.worker}_frame_{index}.png')
                if index == 0:
                    image.save(args.output / f'{args.worker}.png')
    times = [r['ms'] for r in records]
    model_info = {}
    for name in (det, rec):
        directory = Path.home() / '.paddlex' / 'official_models' / name
        model_info[name] = {p.name: p.stat().st_size for p in directory.iterdir() if p.is_file()}
        import yaml
        config = yaml.safe_load((directory / 'inference.yml').read_text())
        post = config.get('PostProcess', {})
        chars = post.get('character_dict', [])
        model_info[name]['postprocess'] = post.get('name')
        model_info[name]['dictionary_length'] = len(chars)
        model_info[name]['dictionary_has_chinese'] = any('\u4e00' <= c <= '\u9fff' for s in chars for c in s)
    # Use the public export API (PaddleOCR does not expose get_config).
    import yaml
    config_path = args.output / f'{args.worker}_pipeline.yml'
    ocr.export_paddlex_config_to_yaml(str(config_path))
    pipeline_config = yaml.safe_load(config_path.read_text())
    orientation_config = pipeline_config.get('SubModules', {}).get('TextLineOrientation', {})
    output = dict(model=args.worker, model_names=[det, rec], settings=SETTINGS,
                  pipeline_config=pipeline_config,
                  textline_orientation_model_name=orientation_config.get('model_name'),
                  versions=dict(paddle=paddle.__version__, paddleocr=paddleocr.__version__),
                  model_files=model_info, warmup_frames=len(frames), rounds=args.rounds,
                  mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                  p95_ms=float(np.percentile(times, 95)), records=records)
    (args.output / f'{args.worker}.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(args.worker, output['mean_ms'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=ROOT.parent / 'images/VID20260929150008.mp4')
    parser.add_argument('--output', type=Path, default=ROOT / 'output-benchmark')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--worker', choices=MODELS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    if args.worker:
        worker(args)
        return
    import cv2
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0 or count < 7:
        raise RuntimeError('Cannot read video metadata')
    indices = np.linspace(0, count - 1, 7, dtype=int)
    frames = []
    wanted = set(int(i) for i in indices)
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decoded in wanted:
            i = int(np.where(indices == decoded)[0][0])
            name = f'frame_{i}.png'
            if not cv2.imwrite(str(args.output / name), frame):
                raise RuntimeError('Cannot save frame')
            frames.append(dict(file=name, index=decoded, seconds=float(decoded / fps), shape=list(frame.shape)))
        decoded += 1
    cap.release()
    if len(frames) != len(indices):
        raise RuntimeError(f'Cannot decode selected frames: got {len(frames)} of {len(indices)}')
    (args.output / 'frames.json').write_text(json.dumps(dict(video=str(args.video.resolve()), fps=fps,
                                                          frame_count=count, selection='7 integer linspace indices spanning the full video; not every second',
                                                          frames=frames), indent=2))
    env = os.environ.copy()
    env.pop('LD_LIBRARY_PATH', None)
    for name in MODELS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name,
                        '--output', str(args.output), '--rounds', str(args.rounds)], env=env, check=True)
    results = {name: json.loads((args.output / f'{name}.json').read_text()) for name in MODELS}
    baseline = results['v6_small']['records'][:7]
    for name, result in results.items():
        comparisons = []
        for current, base in zip(result['records'][:7], baseline):
            intersection = Counter(current['numeric_texts']) & Counter(base['numeric_texts'])
            comparisons.append(dict(frame=current['frame'], exact_text_list=current['texts'] == base['texts'],
                                    numeric_multiset_overlap=sum(intersection.values()),
                                    baseline_numeric_count=len(base['numeric_texts']),
                                    current_numeric_count=len(current['numeric_texts'])))
        result['agreement_vs_v6_small_not_accuracy'] = comparisons
    (args.output / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({n: {k: r[k] for k in ('mean_ms', 'median_ms', 'p95_ms')} for n, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
