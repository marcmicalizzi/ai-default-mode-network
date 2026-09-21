"""Separately reviewed adoption of a completed, previously unadopted candidate."""
from __future__ import annotations
import copy

from .deep_sleep import read_run
from .sleep_plans import read_record, seal, identity, implementation_identity
from .learning import read_plan
from .training import GPU_KIND


def original(store, run_id, parent, instance_id):
    run = read_run(store, run_id)
    compiled = read_record(store, 'sleep_executions', run['execution'])
    if (run['phase'] != 'WakeCommitted' or run['report'].get('outcome') != 'review_candidate_under_original_weights' or
            compiled['recipe']['kind'] != GPU_KIND or compiled.get('candidate_reuse') or
            compiled['parent'] != parent or compiled['instance_id'] != instance_id or
            compiled['preferences']['adoption'] != 'review_first'):
        raise ValueError('candidate is not a completed review-first result for these unchanged parent weights')
    if read_plan(store, compiled['draft_revision'])['status'] != 'draft':
        raise ValueError('candidate source draft was withdrawn or superseded')
    return run, compiled


def prepare(runtime, run_id):
    run, old = original(runtime.store, run_id, identity(runtime.backend.fingerprint), runtime.state['instance_id'])
    # Revalidate the actual factors and process receipts before presenting an
    # adoption plan. Preparation never changes weights or supplies approval.
    from .training_executor import TrainingExecutor
    candidate = TrainingExecutor(runtime.root / 'sleep' / run_id).recover(runtime.root, old)
    if candidate != run['candidate']:
        raise ValueError('review candidate differs from the completed receipt')
    value = copy.deepcopy({k: v for k, v in old.items() if k != 'revision'})
    value.update(implementation=implementation_identity(), training_requested=False,
        candidate_reuse={'run_id': run_id, 'execution': old['revision'], 'receipt': candidate['receipt']},
        adapter_operation='adopt_exact_completed_candidate; no training; rebuild current retained context',
        limitation='No additional training. Candidate was trained earlier; its effects are not guaranteed beneficial. '
                   'The new sleep boundary preserves current retained tokens and sampler state.',
        preferences={**value['preferences'], 'adoption': 'automatic_if_checks_pass'})
    value['training'] = {**value['training'], 'performed_in_prior_run': run_id, 'additional_steps': 0}
    return seal(value)


def recover(root, store, compiled):
    reuse = compiled['candidate_reuse']
    run, old = original(store, reuse['run_id'], compiled['parent'], compiled['instance_id'])
    if old['revision'] != reuse['execution'] or run['candidate']['receipt'] != reuse['receipt']:
        raise ValueError('candidate adoption identity changed')
    from .training_executor import TrainingExecutor
    candidate = TrainingExecutor(root / 'sleep' / run['id']).recover(root, old)
    if candidate != run['candidate']:
        raise ValueError('candidate adoption receipt changed')
    return {**candidate, 'training_performed': False, 'training_in_prior_run': run['id']}
