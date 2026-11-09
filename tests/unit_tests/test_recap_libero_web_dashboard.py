"""Read-only RECAP web server smoke tests (no policy model or GPU required)."""
from __future__ import annotations

import csv
import json
import tempfile
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from toolkits.recap_libero_web_dashboard import (
    Experiment, Handler, ThreadingHTTPServer, TrainingSidecar, load_experiment,
    read_value_curve,
)


def _fixture(tmp_path: Path):
    output = tmp_path / "run"
    (output / "baseline").mkdir(parents=True)
    (output / "recap").mkdir(parents=True)
    (output / "values").mkdir(parents=True)
    (output / "run_config.json").write_text(json.dumps({"episodes": 2, "seed": 7, "value_observations": True}))
    rows = []
    for ep in (1, 2):
        row = {"episode": ep, "seed": 6 + ep}
        for policy in ("baseline", "recap"):
            good = (ep == 1) if policy == "recap" else False
            file = f"{policy}/task0_episode{ep}_{'success' if good else 'failure'}.mp4"
            (output / file).write_bytes(b"\x00\x00\x00\x18ftypisomFAKE-VIDEO-BYTES")
            row[f"{policy}_success"] = int(good)
            row[f"{policy}_video"] = file
            with (output / "values" / f"{policy}_episode{ep:02d}.csv").open('w', newline='') as f:
                w = csv.DictWriter(f,fieldnames=["episode","seed","policy","frame_index","sim_step","value"])
                w.writeheader()
                for i in range(3):w.writerow({"episode":ep,"seed":ep+6,"policy":policy,"frame_index":i,"sim_step":10+10*i,"value":-0.5+i*0.1})
        rows.append(row)
    with (output / "paired_results.csv").open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    return output


def test_valid_experiment(tmp_path):
    output = _fixture(tmp_path)
    manifest, rows = load_experiment(output, expected_episodes=2, expected_seed=7)
    assert manifest['episodes']==2 and len(rows)==2
    e = Experiment(output, 2, 7, output / "not-present.csv", None)
    assert e.summary()['recap_successes']==1
    assert e.episode(1)['baseline']['values']==[-.5,-.4,-.3]
    assert e.episode(1)['recap']['url']=='/media/recap/1'


def test_reject_path_traversal(tmp_path):
    output = _fixture(tmp_path)
    csv_path=output/'paired_results.csv'
    s=csv_path.read_text().replace('baseline/task0_episode1_failure.mp4','../private.mp4')
    csv_path.write_text(s)
    with pytest.raises(ValueError,match='Unsafe'):
        load_experiment(output, expected_episodes=2)


def test_reject_bad_curve(tmp_path):
    output = _fixture(tmp_path)
    curve=output / 'values/baseline_episode01.csv'
    assert read_value_curve(curve, 1, 7, 'baseline') == [-.5,-.4,-.3]
    s=curve.read_text().replace('baseline','recap')
    curve.write_text(s)
    with pytest.raises(ValueError,match='metadata'):
        read_value_curve(curve,1,7,'baseline')


def test_http_ranges_and_api(tmp_path):
    e=Experiment(_fixture(tmp_path),2,7,tmp_path/'missing.csv',None)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.experiment=e
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    host=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(host+'/api/summary') as r:
            assert json.loads(r.read())['episodes']==2
        with urlopen(host+'/api/episode?episode=1') as r:
            assert json.loads(r.read())['baseline']['values'][-1]==-.3
        with urlopen(Request(host+'/media/recap/1',headers={'Range':'bytes=3-8'})) as r:
            assert r.status==206 and r.read()==b'\x18ftypi'
            assert r.headers['Accept-Ranges']=='bytes'
        with urlopen(Request(host+'/media/recap/1',method='HEAD')) as r:
            assert r.status==200 and int(r.headers['Content-Length'])>0
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(host+'/media/recap/1',headers={'Range':'bytes=9999-'}))
        assert exc.value.code==416
    finally:
        server.shutdown();server.server_close();thread.join(2)
