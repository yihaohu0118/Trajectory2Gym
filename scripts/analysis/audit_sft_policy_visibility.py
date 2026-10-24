#!/usr/bin/env python3
"""Audit visibility only; do not silently modify any frozen training view."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from trajectory2gym.training.policy_view import policy_visible_projection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    counts=Counter();by_source={};examples=[];witness=None
    for line in a.input.open():
        row=json.loads(line);messages=row.get('messages') or []
        source=row.get('source_corpus','unknown');domain=row.get('source_domain','unknown')
        key=source+'/'+domain;stats=by_source.setdefault(key,Counter());stats['rows']+=1;counts['rows']+=1
        visible,receipt=policy_visible_projection(messages)
        removed=receipt['removed_messages']
        if removed:
            stats['affected_rows']+=1;counts['affected_rows']+=1
            for item in removed:stats[item['reason']]+=1;counts[item['reason']]+=1
            if len(examples)<8:examples.append({'trace_hash':row.get('trace_hash'),'source_task_id':row.get('source_task_id'),'projection':receipt})
            if witness is None and not row.get('preformatted_prompt'):
                witness=(row,visible,receipt)
    result={'format':'trajectory2gym_sft_policy_visibility_audit_v1','status':'audit_complete',
            'input':str(a.input),'counts':dict(counts),'by_source':{k:dict(v) for k,v in by_source.items()},
            'examples':examples,'training_data_modified':False,'benchmark_performance_causality_proven':False}
    if a.tokenizer and witness:
        from transformers import AutoTokenizer
        from scripts.train.train_trajectory2gym_sft_view import row_to_text,assistant_only_tokenized_row
        tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
        row,visible,receipt=witness;projected={**row,'messages':visible}
        old_text=row_to_text(row,tokenizer);new_text=row_to_text(projected,tokenizer)
        private_contents=[row['messages'][entry['message_index']].get('content') for entry in receipt['removed_messages']]
        private_contents=[x for x in private_contents if isinstance(x,str) and x]
        old=assistant_only_tokenized_row(row,tokenizer,max_length=163840)
        new=assistant_only_tokenized_row(projected,tokenizer,max_length=163840)
        labels_old=old['labels'][old['labels']!=-100].tolist();labels_new=new['labels'][new['labels']!=-100].tolist()
        result['tokenizer_witness']={'trace_hash':row.get('trace_hash'),'tokenizer':str(a.tokenizer),
            'actual_training_renderer':'scripts.train.train_trajectory2gym_sft_view',
            'private_strings_present_in_current_input':sum(text in old_text for text in private_contents),
            'private_strings_removed_from_input':sum(text in old_text and text not in new_text for text in private_contents),
            'supervised_token_sequence_unchanged':labels_old==labels_new,
            'old_input_tokens':len(old['input_ids']),'projected_input_tokens':len(new['input_ids']),
            'old_target_tokens':len(labels_old),'projected_target_tokens':len(labels_new)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},sort_keys=True))


if __name__=='__main__':main()
