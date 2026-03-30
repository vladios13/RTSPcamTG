import pprint
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sanic import Sanic, response
from sanic.response import json, html
import cv2
import json as json_lib
from threading import Timer

from app import notifier
from app import stream
from app import state
from app.utils import zero_division


env = Environment(
    loader=FileSystemLoader('templates'),
    autoescape=select_autoescape(['html', 'xml', 'tpl', 'j2'])
)


def template(tpl, **kwargs):
    t = env.get_template(tpl)
    return html(t.render(kwargs))


app = Sanic('camTGrtsp')

app.static('/static', './static')
app.static('/alarm-files', './alarm')


@app.route('/')
async def mainList(request):
    filtered = {k: v for k, v in state.framebuffer.items() if '_' not in k}
    return template(
        'index.html',
        images=filtered,
        processed=state.get_counter('images_processed'),
        skipped=state.get_counter('images_skipped'),
        avg=zero_division(state.get_counter('images_time'), state.get_counter('images_processed')),
        skip_avg=zero_division(state.get_counter('skipped_time'), state.get_counter('images_skipped')),
        diff_avg=zero_division(state.get_counter('total_skip_diff'), state.get_counter('images_skipped')),
        diff_avg2=zero_division(state.get_counter('total_diff'), state.get_counter('total_processed')),
        total=state.get_counter('images_time'),
        stream_resets=state.get_counter('stream_resets'),
        connect_failures=state.get_counter('stream_connect_failures'),
        size=state.get_size(),
    )


@app.route('/video/<tag>')
async def videoPage(request, tag):
    return template('video.html', tag=tag)


@app.route('/config')
async def configPage(request):
    return template('config.html')


@app.route('/config.json')
async def configJson(request):
    return response.json(state.config)


@app.post('/config.save')
async def configSave(request):
    bb = request.json
    pprint.pprint(bb)
    state.config = bb

    state.stopStreams = True

    timer = Timer(5, stream.loadStreams)
    timer.start()

    notifier.stop()
    notifier.initBot()
    notifier.begin()

    with open('config.json', 'w') as file:
        file.write(json_lib.dumps(state.config, indent=2, ensure_ascii=False))

    return response.json(bb)


@app.route('/favicon.ico')
async def favicon(request):
    return response.empty(status=204)


@app.route('/snapshot/<tag>')
async def snapshot(request, tag):
    if tag in state.framebuffer:
        frame = state.framebuffer[tag].copy()
        for s in state.config['streams']:
            if s['label'] == tag and s.get('detect_in_polygon'):
                ctr = np.array(s['detect_in_polygon']).reshape((-1, 1, 2)).astype(np.int32)
                cv2.drawContours(frame, [ctr], -1, (0, 255, 0), 3)

        _, jpg = cv2.imencode('.jpg', frame)
        return response.raw(jpg, content_type='image/jpeg', headers={'Cache-Control': 'no-store'})
    else:
        return response.html('No image')


@app.route('/snapshot/raw/<tag>')
async def snapshot_raw(request, tag):
    if tag in state.framebuffer:
        frame = state.framebuffer[tag].copy()
        h, w = frame.shape[:2]
        _, jpg = cv2.imencode('.jpg', frame)
        return response.raw(jpg, content_type='image/jpeg', headers={
            'Cache-Control': 'no-store',
            'X-Frame-Width': str(w),
            'X-Frame-Height': str(h),
        })
    else:
        return response.html('No image', status=404)


@app.route('/stats')
async def stats(request):
    return response.html(pprint.pformat(state.frame_stats))


@app.route('/api/stats')
async def api_stats(request):
    return response.json({
        'processed': state.get_counter('images_processed'),
        'skipped': state.get_counter('images_skipped'),
        'avg': zero_division(state.get_counter('images_time'), state.get_counter('images_processed')),
        'skip_avg': zero_division(state.get_counter('skipped_time'), state.get_counter('images_skipped')),
        'diff_avg': zero_division(state.get_counter('total_skip_diff'), state.get_counter('images_skipped')),
        'diff_avg2': zero_division(state.get_counter('total_diff'), state.get_counter('total_processed')),
        'total': state.get_counter('images_time'),
        'stream_resets': state.get_counter('stream_resets'),
        'connect_failures': state.get_counter('stream_connect_failures'),
        'size': state.get_size(),
        'streams': list({k: None for k in state.framebuffer if '_' not in k}.keys()),
    })


@app.route('/about')
async def about_page(request):
    return template('about.html')


@app.route('/movement')
async def movement_page(request):
    return template('movement.html')


@app.route('/api/movement')
async def api_movement(request):
    result = {}
    for cam, entries in state.frame_stats.items():
        # Downsample: max 300 points per camera
        step = max(1, len(entries) // 300)
        sampled = entries[::step]
        result[cam] = [
            {'t': round(e['time'] * 1000), 'v': round(e['stat'] * 1000, 3)}
            for e in sampled
        ]
    return response.json(result)


@app.route('/api/alarms')
async def api_alarms(request):
    cam_filter = request.args.get('cam', 'all')
    try:
        offset = max(0, int(request.args.get('offset', 0)))
        limit = min(200, max(1, int(request.args.get('limit', 50))))
    except ValueError:
        return response.json({'error': 'invalid offset or limit'}, status=400)

    alarm_dir = Path('alarm')
    all_files = []
    cams = []

    if alarm_dir.exists():
        cam_dirs = sorted([d for d in alarm_dir.iterdir() if d.is_dir()])
        cams = [d.name for d in cam_dirs]

        for cam_dir in cam_dirs:
            if cam_filter != 'all' and cam_dir.name != cam_filter:
                continue
            date_dirs = sorted([d for d in cam_dir.iterdir() if d.is_dir()], reverse=True)
            for date_dir in date_dirs:
                for f in sorted(date_dir.iterdir(), reverse=True):
                    if f.suffix.lower() != '.jpg':
                        continue
                    # Разбираем время из имени: 2026-03-23_22_45_08_807018.jpg
                    parts = f.stem.split('_')
                    time_str = f'{parts[1]}:{parts[2]}:{parts[3]}' if len(parts) >= 4 else ''
                    all_files.append({
                        'cam': cam_dir.name,
                        'date': date_dir.name,
                        'time': time_str,
                        'url': f'/alarm-files/{cam_dir.name}/{date_dir.name}/{f.name}',
                        '_sort': f.name,
                    })

    # Сортируем по имени файла (содержит полный timestamp) — новые первые
    all_files.sort(key=lambda x: x['_sort'], reverse=True)
    for item in all_files:
        del item['_sort']

    total = len(all_files)
    return response.json({
        'items': all_files[offset:offset + limit],
        'has_more': (offset + limit) < total,
        'cams': cams,
    })


def begin():
    app.run(host='0.0.0.0', port=8000)
