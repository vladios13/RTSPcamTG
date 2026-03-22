import pprint
import time

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

    state.framebuffer = {}

    notifier.initBot()
    notifier.begin()

    with open('config.json', 'w') as file:
        file.write(json_lib.dumps(state.config))

    return response.json(bb)


@app.route('/favicon.ico')
async def favicon(request):
    return response.empty(status=204)


@app.route('/snapshot/<tag>')
async def snapshot(request, tag):
    if tag in state.framebuffer:
        for s in state.config['streams']:
            if s['label'] == tag and 'detect_in_polygon' in s:
                ctr = np.array(s['detect_in_polygon']).reshape((-1, 1, 2)).astype(np.int32)
                cv2.drawContours(state.framebuffer[tag], [ctr], -1, (0, 255, 0), 3)

        _, jpg = cv2.imencode('.jpg', state.framebuffer[tag])
        return response.raw(jpg, content_type='image/jpeg', headers={'Cache-Control': 'no-store'})
    else:
        return response.html('No image')


@app.route('/stats')
async def stats(request):
    return response.html(pprint.pformat(state.frame_stats))


@app.route('/munin')
async def munin(request):
    result = {}
    for k, v in state.frame_stats.items():
        if len(v) > 1000:
            state.logger.info('Trimming to 1000')
            state.frame_stats[k] = v[-1000:]

    for k, v in state.frame_stats.items():
        frames = 0
        diff = 0
        for stat in v:
            if time.time() - stat['time'] < 60 * 5:
                frames += 1
                diff += stat['stat']

        result[k] = zero_division(diff, frames)

    return template('munin.html', stats=result)


def begin():
    app.run(host='0.0.0.0', port=8000)
