import importlib.util
from pathlib import Path
import pytest
import hashlib
import json


def test_bound_receipt_load_rejects_mutation_and_wrong_run(tmp_path):
    receipt = tmp_path / 'c1.json'
    receipt.write_text(json.dumps(probe()))
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    manifest = {'schema': 'glm53-ab-receipts-v1', 'run_id': 'test-run',
                'source_manifest_sha256': 'a' * 64, 'files': {'c1.json': digest}}
    assert v.load_bound_receipt(tmp_path, manifest, 'c1.json', run_id='test-run', source_hash='a' * 64) == probe()
    with pytest.raises(ValueError):
        v.load_bound_receipt(tmp_path, manifest, 'c1.json', run_id='other-run', source_hash='a' * 64)
    receipt.write_text('{}')
    with pytest.raises(ValueError):
        v.load_bound_receipt(tmp_path, manifest, 'c1.json', run_id='test-run', source_hash='a' * 64)



def test_bound_receipt_rejects_duplicate_json_fields(tmp_path):
    receipt = tmp_path / 'c1.json'
    receipt.write_text('{"speed": 1, "speed": 100}')
    manifest = {'schema': 'glm53-ab-receipts-v1', 'run_id': 'test-run',
                'source_manifest_sha256': 'a' * 64,
                'files': {'c1.json': hashlib.sha256(receipt.read_bytes()).hexdigest()}}
    with pytest.raises(ValueError, match='duplicate'):
        v.load_bound_receipt(tmp_path, manifest, 'c1.json', run_id='test-run', source_hash='a' * 64)


def probe(speed=46.0):
    rows = [{'index': i, 'kind': kind, 'completion_tokens': 512,
             'decode_tok_s': speed, 'decode_seconds': round(512 / speed, 3),
             'metric_delta': {'drafts': 280, 'draft_tokens': 280, 'accepted_tokens': 232}}
            for i, kind in enumerate(('prose', 'prose', 'code', 'code'))]
    return {'schema': 'glm53-dflash2-uva-acceptance-v1', 'model': 'glm-5.3-big',
            'max_tokens': 512, 'rows': rows,
            'summary': {'requests': 4, 'completion_tokens': 2048,
                        'verification_steps': 1120,
                        'acceptance_length_weighted': 2048 / 1120,
                        'decode_tok_s_median': speed}}


def test_validate_lane_derives_speed_and_checks_raw_counter_consistency():
    p = probe()
    assert v.validate_lane(p) == 46.0
    p['summary']['decode_tok_s_median'] = 999
    with pytest.raises(ValueError):
        v.validate_lane(p)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, 0, True, '46'])
def test_validate_lane_rejects_invalid_raw_speed(bad):
    p = probe()
    p['rows'][0]['decode_tok_s'] = bad
    with pytest.raises(ValueError):
        v.validate_lane(p)


@pytest.mark.parametrize('field,value', [('drafts', 0), ('draft_tokens', 1), ('accepted_tokens', 999), ('drafts', 1.5), ('accepted_tokens', True)])
def test_validate_lane_rejects_impossible_counters(field, value):
    p = probe()
    p['rows'][0]['metric_delta'][field] = value
    with pytest.raises(ValueError):
        v.validate_lane(p)

SCRIPT = Path(__file__).resolve().parents[1] / 'recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/scripts/window_ab_verdict.py'
spec = importlib.util.spec_from_file_location('ab_verdict_test', SCRIPT)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def test_published_real_k1_probe_is_accepted_without_reinterpreting_duration():
    path = SCRIPT.parents[1] / 'results/2026-09-08-e1-v2-live/live-receipts/k1-acceptance.json'
    payload = json.loads(path.read_text())
    assert v.validate_lane(payload) == payload['summary']['decode_tok_s_median']


def test_c1_gate_blocks_regression_or_quality_mismatch():
    reference = {str(i): f'answer {i}' for i in range(20)}
    assert v.c1_gate(probe(), reference, reference)['pass'] is True
    assert v.c1_gate(probe(44), reference, reference)['pass'] is False
    changed = {**reference, '0': 'wrong'}
    assert v.c1_gate(probe(), reference, changed)['pass'] is False


def test_evaluate_reports_measured_gates_not_release_approval():
    q = {str(i): f'answer {i}' for i in range(20)}
    result = v.evaluate(probe(46), probe(49), q, q, q)
    assert result['measurement_gates_pass'] is True
    assert result['verdict'] == 'INCONCLUSIVE'
    assert result['promotion_authorized'] is False
    assert v.evaluate(probe(46), probe(47), q, q, q)['measurement_gates_pass'] is False
    assert v.evaluate(probe(46), probe(49), q, q, {**q, '0': 'wrong'})['measurement_gates_pass'] is False
    broken = probe()
    broken['rows'] = []
    result = v.evaluate(broken, probe(49), q, q, q)
    assert result['verdict'] == 'INCONCLUSIVE'
    assert result['measurement_gates_pass'] is False
    assert result['error']


def test_quality_requires_all_twenty_nonempty_outputs_and_exact_equality():
    reference = {str(i): f'answer {i}' for i in range(20)}
    assert v.quality_match(reference, dict(reference)) is True
    candidate = dict(reference)
    candidate['19'] = 'different'
    assert v.quality_match(reference, candidate) is False
    for invalid in ({}, {**reference, '20': 'extra'}, {**reference, '0': ''}, {**reference, '0': '\u241f'}):
        with pytest.raises(ValueError):
            v.quality_match(invalid, reference)
        with pytest.raises(ValueError):
            v.quality_match(reference, invalid)
