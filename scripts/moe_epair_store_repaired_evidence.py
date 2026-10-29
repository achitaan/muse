"""Bind the namespace-only repair; reuse all unchanged scientific gates."""
import hashlib
import json
import moe_epair_store_evidence as original
from moe_epair_store_evidence import ROOT, PARENT, CUDA, SHAPES, full_result, verify_trial

FAILED = original.TRIAL
TRIAL = '9fc2993a9cc3b3723ad24c8016cca6bf9915d53958b821da222f5ebe240d944d'
PREFLIGHT = ROOT / 'results/moe-epair-store-repair-20261029'


def read(path):
    return json.loads(path.read_bytes())


def directory(sha):
    return ROOT / '.moe-epair-measurements' / sha


def source():
    raw = (ROOT / 'candidates/moe_active.py').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT, TRIAL), ('Unreviewed source', sha)
    return raw, sha


def bound_result(sha):
    if sha == PARENT:
        return original.bound_result(sha)
    folder = directory(sha)
    assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == sha
    launch, result = read(folder / 'launch.json'), read(folder / 'result.json')
    assert launch['source_sha256'] == sha
    assert result['id'] == launch.get('resolved_job_id', launch['job_id'])
    return result


def validate_packet():
    failed = original.validate_packet()
    assert hashlib.sha256(failed).hexdigest() == FAILED
    raw = (PREFLIGHT / 'submission.py').read_bytes()
    assert failed.count(b'okc_epair_y::') == 9
    assert raw == failed.replace(b'okc_epair_y::', b'okc_epair::')
    assert hashlib.sha256(raw).hexdigest() == TRIAL
    audit = read(PREFLIGHT / 'query-repair-audit.json')
    assert audit['failed_source_sha256'] == FAILED and audit['source_sha256'] == TRIAL
    assert audit['cuda_sha256'] == CUDA and audit['query_namespace_references_corrected'] == 9
    assert audit['only_query_constant_AST_changed'] and audit['every_runtime_function_AST_unchanged']
    assert audit['same_resource_query_as_previously_compiled_factor_module_after_identifier_binding']
    assert audit['exports_declarations_and_prepare_method_agree']
    assert set(audit['rejected_mutants']) == {'original_wrong_namespace', 'wrong_geometry_namespace',
                                             'missing_export', 'wrong_host_method'}
    assert audit['numerical_resource_selection_and_performance_gates_unchanged']
    return raw
