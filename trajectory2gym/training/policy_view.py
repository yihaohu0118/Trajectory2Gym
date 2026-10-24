"""Actor-visible projection of a full interaction log, retaining raw evidence elsewhere.

This module does not change target actions or select trajectories. It uses only
message roles and recorded tool-call ownership, never benchmark/category names.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence


def policy_visible_projection(messages: Sequence[Mapping[str,Any]]) -> tuple[list[dict],dict]:
    owners: dict[str,str]={}
    kept=[];removed=[];assistant_before=0
    for index,message in enumerate(messages):
        role=message.get('role')
        calls=message.get('tool_calls') or []
        if role in ('assistant','user'):
            for call in calls:
                call_id=call.get('id')
                if call_id is not None:
                    key=str(call_id)
                    if key in owners and owners[key]!=role:
                        raise ValueError('ambiguous tool-call ownership')
                    owners[key]=role
        if role=='assistant':assistant_before+=1
        reason=None
        if role=='user' and calls:
            reason='private_external_actor_call'
        elif role=='tool':
            explicit=message.get('requestor')
            # Call IDs are scoped to outstanding calls, not the whole episode.
            # Some valid transports reuse an ID after its response arrives.
            inferred=owners.pop(str(message.get('tool_call_id')),None)
            if explicit in ('assistant','user') and inferred and explicit!=inferred:
                raise ValueError('tool observation ownership contradicts its call')
            if explicit=='user' or inferred=='user':
                reason='private_external_actor_observation'
        if reason:
            removed.append({'message_index':index,'reason':reason,
                            'tool_call_id':message.get('tool_call_id')})
        else:
            kept.append(copy.deepcopy(dict(message)))
    if sum(m.get('role')=='assistant' for m in kept)!=assistant_before:
        raise AssertionError('projection may not delete any assistant decision')
    return kept,{'format':'trajectory2gym_actor_visible_projection_v1',
        'input_messages':len(messages),'output_messages':len(kept),'removed_messages':removed,
        'assistant_turns_unchanged':True,'benchmark_features_used':False,
        'raw_evidence_must_remain_preserved':True}


def project_sft_row(row: Mapping[str,Any]) -> tuple[dict,dict]:
    """Separate causal policy context from preserved full-event evidence."""
    if 'evidence_messages' in row:
        evidence_message_history(row)  # validate an already projected row
        return copy.deepcopy(dict(row)),dict(row.get('policy_visibility_projection') or {})
    messages=row.get('messages') or []
    visible,receipt=policy_visible_projection(messages)
    if not receipt['removed_messages']:
        return copy.deepcopy(dict(row)),receipt
    if 'preformatted_prompt' in row or 'prompt' in row or 'response' in row:
        raise ValueError('preformatted/decision rows require their explicit native transport projector')
    original_assistant=[m for m in messages if m.get('role')=='assistant']
    if original_assistant != [m for m in visible if m.get('role')=='assistant']:
        raise ValueError('policy projection changed assistant decisions')
    result=copy.deepcopy(dict(row))
    result['messages']=visible
    result['evidence_messages']=copy.deepcopy(messages)
    receipt={**receipt,'evidence_messages_sha256':hashlib.sha256(
        json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'context_hashes_from_source_are_not_reusable':True,
        'fresh_nll_required_for_changed_context':True}
    result['policy_visibility_projection']=receipt
    # Old exact-context certificates remain provenance, not certificates for
    # the newly rendered policy context. SFT observation/action evidence stays
    # intact in evidence_messages and action-only certificate fields.
    if 'state_certificates_by_assistant_turn' in result:
        result['evidence_state_certificates_by_assistant_turn']=result.pop('state_certificates_by_assistant_turn')
    return result,receipt


def evidence_message_history(row: Mapping[str,Any]) -> list[dict]:
    """Read completion evidence without allowing phantom assistant actions."""
    messages=list(row.get('messages') or [])
    evidence=row.get('evidence_messages')
    if evidence is None:return messages
    if not isinstance(evidence,list):raise ValueError('evidence_messages must be a list')
    receipt=row.get('policy_visibility_projection') or {}
    digest=hashlib.sha256(json.dumps(evidence,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if receipt.get('format')!='trajectory2gym_actor_visible_projection_v1' or receipt.get('evidence_messages_sha256')!=digest:
        raise ValueError('full-event evidence is not bound to the projection')
    visible,_=policy_visible_projection(evidence)
    if visible!=messages:
        raise ValueError('policy messages disagree with their full-event evidence')
    return list(evidence)
