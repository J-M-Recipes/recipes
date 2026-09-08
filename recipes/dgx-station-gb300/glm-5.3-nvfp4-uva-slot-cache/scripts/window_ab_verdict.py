"""Fail-closed offline gates for the matched K1/K2 window."""

import math
import statistics
import hashlib
import json
import re
from pathlib import Path


def load_bound_receipt(root, manifest, relative, *, run_id, source_hash):
    """Load bytes once and bind them to a trusted caller-supplied manifest.

    Hash binding is integrity, not proof that an instrument produced the bytes.
    The caller must obtain this manifest from the current run, not the receipt.
    """
    if not isinstance(manifest, dict) or manifest.get('schema') != 'glm53-ab-receipts-v1':
        raise ValueError('receipt manifest schema mismatch')
    if not isinstance(run_id, str) or not run_id or manifest.get('run_id') != run_id:
        raise ValueError('receipt run mismatch')
    if not isinstance(source_hash, str) or not re.fullmatch('[0-9a-f]{64}', source_hash) or manifest.get('source_manifest_sha256') != source_hash:
        raise ValueError('receipt source mismatch')
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or str(rel) != relative:
        raise ValueError('invalid receipt path')
    root = Path(root).resolve()
    path = root / rel
    if not path.resolve().is_relative_to(root) or any(part.is_symlink() for part in [path, *path.parents] if part != root):
        raise ValueError('receipt symlink or path escape')
    files = manifest.get('files')
    expected = files.get(relative) if isinstance(files, dict) else None
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('receipt hash missing or invalid')
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError('receipt unavailable') from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('receipt hash mismatch')
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique_fields)


def _number(value, *, minimum=0, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError('invalid numeric measurement')
    if integer and int(value) != value:
        raise ValueError('counter must be integral')
    return value


def validate_lane(payload: dict) -> float:
    """Validate the existing 512-token acceptance receipt, return row median.

    This is a client-observed throughput metric, NOT a GPU decode duration.
    Millisecond rounding of historical row durations is explicitly allowed.
    """
    if not isinstance(payload, dict) or payload.get('schema') != 'glm53-dflash2-uva-acceptance-v1':
        raise ValueError('probe schema mismatch')
    if payload.get('model') != 'glm-5.3-big' or payload.get('max_tokens') != 512:
        raise ValueError('probe configuration mismatch')
    rows, summary = payload.get('rows'), payload.get('summary')
    if not isinstance(rows, list) or len(rows) != 4 or not isinstance(summary, dict):
        raise ValueError('four probe rows and summary required')
    speeds, tokens, steps = [], 0, 0
    for i, (row, kind) in enumerate(zip(rows, ('prose', 'prose', 'code', 'code'))):
        if not isinstance(row, dict) or type(row.get('index')) is not int or row['index'] != i or row.get('kind') != kind:
            raise ValueError('probe row provenance mismatch')
        n = _number(row.get('completion_tokens'), minimum=1, integer=True)
        if n > 512:
            raise ValueError('completion exceeds request limit')
        duration = _number(row.get('decode_seconds'), minimum=0.001)
        speed = _number(row.get('decode_tok_s'), minimum=0.001)
        if not (n / (duration + 0.0005) - 0.0005 <= speed <= n / (duration - 0.0005) + 0.0005):
            raise ValueError('speed inconsistent with rounded row duration')
        metrics = row.get('metric_delta')
        if not isinstance(metrics, dict):
            raise ValueError('missing raw counters')
        drafts = _number(metrics.get('drafts'), minimum=1, integer=True)
        drafted = _number(metrics.get('draft_tokens'), minimum=1, integer=True)
        accepted = _number(metrics.get('accepted_tokens'), integer=True)
        if accepted > drafted or drafted < drafts:
            raise ValueError('impossible speculative counters')
        speeds.append(speed)
        tokens += n
        steps += drafts
    expected = {'requests': 4, 'completion_tokens': tokens, 'verification_steps': steps,
                'acceptance_length_weighted': tokens / steps,
                'decode_tok_s_median': statistics.median(speeds)}
    for field, value in expected.items():
        observed = _number(summary.get(field))
        if not math.isclose(observed, value, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f'probe summary mismatch: {field}')
    return expected['decode_tok_s_median']


def c1_gate(probe: dict, reference: dict, candidate: dict, baseline_speed=45.65) -> dict:
    """Cheap pre-C2 gate; historical baseline is a hygiene reference only."""
    baseline = _number(baseline_speed, minimum=0.001)
    speed = validate_lane(probe)
    quality = quality_match(reference, candidate)
    return {'pass': speed >= baseline * 0.99 and quality,
            'speed': speed, 'quality_pass': quality, 'baseline_speed': baseline}


def evaluate(c1_probe, c2_probe, reference, c1_quality, c2_quality) -> dict:
    """Report measurement gates only: no bound run manifest or release supplied.

    The lifecycle runner must independently verify source, receipt provenance,
    restoration, and profiling prerequisites. This helper cannot authorize PASS.
    """
    result = {'schema': 'glm53-ab-measurement-gates-v1', 'verdict': 'INCONCLUSIVE',
              'promotion_authorized': False, 'measurement_gates_pass': False,
              'reason': 'measurement-only evaluation; lifecycle/evidence qualification required'}
    try:
        g1 = c1_gate(c1_probe, reference, c1_quality)
        speed2 = validate_lane(c2_probe)
        g2 = speed2 >= g1['speed'] * 1.05
        g3 = quality_match(reference, c2_quality)
        result.update({'g1': g1, 'g2_pass': g2, 'g3_pass': g3,
                       'speed_ratio': speed2 / g1['speed'],
                       'measurement_gates_pass': g1['pass'] and g2 and g3})
    except (ValueError, TypeError, KeyError) as exc:
        result['error'] = str(exc)
    return result


def quality_match(reference: dict, candidate: dict) -> bool:
    expected = {str(i) for i in range(20)}
    for outputs in (reference, candidate):
        if not isinstance(outputs, dict) or set(outputs) != expected:
            raise ValueError('quality requires exactly prompts 0 through 19')
        if any(not isinstance(text, str) or not text.replace('\u241f', '').strip()
               for text in outputs.values()):
            raise ValueError('quality outputs must contain nonempty text')
    return reference == candidate
