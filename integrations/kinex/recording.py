"""Decode required streams and check acquisition timestamps, not encoded FPS."""

import json
import math
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify(workspace, requested, *, minimum_samples=3, minimum_rate_fraction=.5):
    import cv2

    manifests = list((Path(workspace) / '.kinex/recordings').glob('*/recording.json'))
    if len(manifests) != 1:
        raise ValueError('Expected exactly one recording for a fresh trial')
    path = manifests[0]
    record = json.loads(path.read_text())
    if record['status'] != 'stopped' or record.get('error') or record.get('quotaHit'):
        raise ValueError('Recording did not stop cleanly')
    report = {}
    for name, setting in requested.items():
        stream = record['devices'][name].get('stream', {})
        if stream.get('status') != 'stopped':
            raise ValueError(f'Required stream {name} did not stop')
        if 'video' in stream:
            timestamps = rows(path.parent / stream['timestamps'])
            source = sorted(set(item['source_t_ns'] for item in timestamps))
            if any(a['t_ns'] >= b['t_ns'] for a,b in zip(timestamps,timestamps[1:])):
                raise ValueError('Recording presentation timestamps are not ordered')
            videos = ['video'] + (['depth'] if 'depth' in stream else [])
            decoded = {}
            for kind in videos:
                capture = cv2.VideoCapture(str(path.parent / stream[kind]))
                count = 0
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        if kind == 'video' and (frame.max() == 0 or frame.std() < 1):
                            raise ValueError(f'{name} has black or unreadable frames')
                        count += 1
                finally:
                    capture.release()
                if count != len(timestamps) or count < minimum_samples:
                    raise ValueError(f'{name} decoded frame count differs from its timestamps')
                decoded[kind] = count
            report[name] = {'decoded_frames': decoded}
        elif 'state' in stream:
            samples = rows(path.parent / stream['state'])
            source = sorted(set(item['t_ns'] for item in samples))
            if not all(isinstance(item.get('velocities'),list) for item in samples):
                raise ValueError('Missing measured state samples')
            report[name] = {'state_rows': len(samples)}
        else:
            raise ValueError(f'{name} has no supported recorded media')
        if len(source) < minimum_samples:
            raise ValueError(f'{name} lacks independent acquisitions')
        rate = (len(source)-1)*1e9/(source[-1]-source[0])
        if not math.isfinite(rate) or rate < setting['rate'] * minimum_rate_fraction:
            raise ValueError(f'{name} acquired {rate:.3f} Hz below the declared minimum')
        report[name].update(acquisitions=len(source), measured_hz=rate,
                            requested_hz=setting['rate'], minimum_fraction=minimum_rate_fraction)
    return {'passed': True, 'streams': report}
