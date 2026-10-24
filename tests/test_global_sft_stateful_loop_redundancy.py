from trajectory2gym.training.global_sft_selection import (
    _certified_exact_adjacent_loop_redundancy,
    prepare_candidates,
)


def row(observations, action_only=()):
    messages=[]
    for observation in observations:
        messages.extend([
            {"role":"assistant","tool_calls":[{"name":"get_balance","arguments":{"account_id":"x"}}]},
            {"role":"tool","content":observation},
        ])
    return {"trace_hash":"h","source_task_id":"t","trajectory_family_id":"f","success":True,
        "enable_thinking":False,"messages":messages,"supervised_assistant_turn_indices":list(range(len(observations))),
        "verified_assistant_turn_indices":list(action_only)}


def test_changed_authentic_observation_is_not_a_loop():
    item=row(['{"balance": 1}','{"balance": 2}'])
    assert _certified_exact_adjacent_loop_redundancy(item)==0
    candidates,audit=prepare_candidates([item],supervised_tokens={'h':10},loop_redundancy_mode='certified_exact_adjacent_v2')
    assert candidates[0].loop_redundancy==0
    assert audit['loop_redundancy_mode']=='certified_exact_adjacent_v2'


def test_exact_adjacent_action_and_observation_is_redundant():
    assert _certified_exact_adjacent_loop_redundancy(row(['same','same']))==0.5


def test_action_only_certificate_never_fabricates_duplicate_transition():
    assert _certified_exact_adjacent_loop_redundancy(row(['same','same'],action_only=(0,1)))==0


def test_masked_intervening_decision_breaks_adjacency():
    item=row(['same','same','same']);item['supervised_assistant_turn_indices']=[0,2]
    assert _certified_exact_adjacent_loop_redundancy(item)==0
