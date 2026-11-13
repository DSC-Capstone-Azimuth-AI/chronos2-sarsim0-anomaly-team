"""Offline schedule policy behavior and reviewed-denominator invariants."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from sarsim_fm.anomaly.alert_policy import apply_policy, occurrences, verify_policy_bundle
from sarsim_fm.anomaly.alert_policy_config import validate_policy
from sarsim_fm.anomaly.derconnect_artifacts import BundleReader
from sarsim_fm.anomaly.export import VERSION as RAW_VERSION
from sarsim_fm.anomaly.export import hourly_summary, write_json, write_table
from sarsim_fm.anomaly.policy_contract import contract, schema
from sarsim_fm.anomaly.provenance import file_hash
from sarsim_fm.cli import main
from sarsim_fm.config import load_config

UTC = lambda x: pd.Timestamp(x, tz="UTC")


def make_bundle(root, *, start="2026-03-02 13:50", count=40, intervals=((9, 11),),
                actuals=None, missing=(), split=False, direction=1):
    root.mkdir()
    t = pd.date_range(start, periods=count, freq="min", tz="UTC")
    ident = {"schema_version": RAW_VERSION, "run_id": "source-run", "detector_id": "detector-A",
                 "model_name": "chronos2:synthetic", "meter_id": "E2451", "signal": "total_active_power"}
    rows, events = [], []
    for i, stamp in enumerate(t):
        event_index = next((j for j, (a, b) in enumerate(intervals) if a <= i < b), None)
        member = event_index is not None
        actual = (actuals or {}).get(i, 100 + (10 * direction if member else 0))
        missing_point = i in missing
        rows.append({**ident, "timestamp_utc": stamp, "bin_end_utc": stamp + pd.Timedelta(minutes=1),
                     "available_at_utc": stamp + pd.Timedelta(minutes=1), "forecast_origin_utc": stamp,
                     "horizon_steps": 1, "sampling_seconds": 60, "actual": None if missing_point else actual,
                     "expected_power": 100., "q01": 95., "q10": 98., "q50": 100., "q90": 102., "q99": 105.,
                     "unit": "source_unconfirmed", "score_name": "interval_exceedance",
                     "anomaly_score": None if missing_point else .5 if member or i == 3 else 0.,
                     "threshold": .1, "anomaly_flag": None if missing_point else int(member or i == 3),
                     "event_id": f"event-{event_index}" if member else None, "event_member": member,
                     "anomaly_type": "unknown", "type_source": "not_classified", "prediction_status":
                     "insufficient_coverage" if missing_point else "scored", "observation_count": 0 if missing_point else 30,
                     "coverage_fraction": 0. if missing_point else 1., "quality_flags": '[]', "calibration_id": "cal"})
    for j, (a, b) in enumerate(intervals):
        events.append({**ident, "event_id": f"event-{j}", "start_utc": t[a],
                       "end_utc": t[b-1] + pd.Timedelta(minutes=1), "detected_at_utc": t[a] + pd.Timedelta(minutes=2),
                       "peak_timestamp_utc": t[a], "peak_score": .5, "threshold_at_peak": .1,
                       "duration_seconds": (b-a)*60, "flagged_points": b-a,
                       "anomaly_type": "unknown", "type_source": "not_classified"})
    frame = pd.DataFrame(rows)
    pieces = [frame.iloc[:10], frame.iloc[10:]] if split else [frame]
    names = []
    for i, f in enumerate(pieces):
        name = f"predictions.parquet/E2451/{i}.parquet"
        write_table(root / name, f, "predictions")
        names.append(name)
    write_table(root / 'events.parquet', events, 'events')
    write_table(root / 'quality-events.parquet', [], 'quality-events')
    ev = pd.read_parquet(root / 'events.parquet')
    write_table(root / 'hourly-summary.parquet', hourly_summary(frame, ev), 'hourly-summary')
    write_json(root / 'identity.json', {'config': {'data_kind': 'artificial_fixture'}})
    manifest = {'complete': True, 'run_id': 'source-run', 'detector_id': 'detector-A', 'prediction_parts': names,
                'model': {'spec': 'chronos2:synthetic', 'sha256': 'a'*64, 'source_repository': 'test/model',
                          'resolved_revision': 'a'*40, 'files': {'config.json': 'c'*64}},
                'quantile_kind': 'model_quantiles', 'calibration': {'E2451': {'residual_scale':1.,'scale_points':1440,'calibration_points':10080}}, 'resolved_config': {'meters': ['E2451'], 'detection':
                    {'context_length': 256, 'batch_size': 16, 'min_run_seconds': 120, 'merge_gap_seconds': 360}},
                'outputs': {str(p.relative_to(root)): file_hash(p) for p in root.rglob('*') if p.is_file()}}
    write_json(root / 'run-manifest.json', manifest)
    return root


def policy_for(source, root, *, approved=True, direction="increase", changes=None):
    rule = {"rule_id": "morning", "enabled": True, "meter_id": "E2451", "signal": "total_active_power",
                "weekdays": ['mon','tue','wed','thu','fri','sat','sun'], "local_start": '05:55', "local_end": '06:10',
                "effective_start": '2026-01-01T00:00:00Z', "effective_end": '2027-01-01T00:00:00Z',
                "expected_direction": direction, "preceding_context_minutes": 5, "max_duration_seconds": 300,
                "minimum_absolute_reference": 1., "relative_change_lower": 0. if direction=='increase' else -.5,
                "relative_change_upper": .5 if direction=='increase' else 0., "reference": None,
                "reference_unit": 'source_unconfirmed', "review": {'status': 'approved' if approved else 'pending',
                  'reviewer': 'synthetic_fixture' if approved else None,
                  'reviewed_at': '2026-02-01T00:00:00Z' if approved else None,
                  'schedule_evidence': 'artificial operating schedule' if approved else None}}
    rule.update(changes or {})
    reader = BundleReader(source)
    qa = {'schema_version': 'derconnect-forecast-qa/v1', 'kind': 'derconnect_forecast_qa',
          'source_run_id': reader.manifest['run_id'], 'source_manifest_sha256': file_hash(source/'run-manifest.json'),
          'model_sha256': reader.manifest['model']['sha256'], 'detectors': {'detector-A': {'status': 'pass'}}}
    qa_path = root / 'qa.json'
    write_json(qa_path, qa)
    if approved:
        ref = {'kind': 'derconnect_schedule_reference', 'review_status': 'approved',
               'source_hashes': {'artificial-fixture': 'f'*64}, 'unit': rule['reference_unit'],
               **{k: rule[k] for k in ('rule_id', 'meter_id', 'signal', 'expected_direction',
                                      'minimum_absolute_reference', 'relative_change_lower', 'relative_change_upper',
                                      'preceding_context_minutes', 'max_duration_seconds')},
               **{k: rule['review'][k] for k in ('reviewer','reviewed_at','schedule_evidence')}}
        path = root/'reference.json'
        write_json(path, ref)
        rule['reference'] = {'path': str(path), 'sha256': file_hash(path)}
    return {"kind": 'derconnect_alert_policy',"schema_version": 'derconnect-alert-policy/v1',"policy_name": 'test',
                "timezone": 'America/Los_Angeles',"qa_report": {'path':str(qa_path),'sha256':file_hash(qa_path)},"rules": [rule]}


def read_decisions(out):
    return pd.read_parquet(out/'event-decisions.parquet')


def test_active_preview_and_source_preservation(tmp_path):
    source=make_bundle(tmp_path/'source',split=True,missing=(30,))
    before={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    cfg=policy_for(source,tmp_path)
    preview=apply_policy(source,cfg,tmp_path/'preview')
    active=apply_policy(source,cfg,tmp_path/'active',mode='active')
    assert preview['effective_events']==1 and preview['suppressed_events']==0 and preview['would_suppress_known_true']==1
    assert active['effective_events']==0 and active['suppressed_events']==1
    assert before=={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    h=pd.read_parquet(tmp_path/'active/hourly-policy-summary.parquet')
    assert h.raw_event_starts.sum()==h.suppressed_event_starts.sum()==1
    assert h.has_raw_event_overlap.sum()==2 and h.has_actionable_event_overlap.sum()==0
    manifest=verify_policy_bundle(tmp_path/'active',source)
    p=pd.concat([pd.read_parquet(tmp_path/'active'/n) for n in manifest['point_parts']],ignore_index=True)
    assert p.raw_anomaly_flag.sum()==3 and p.actionable_flag.sum()==0
    assert p.actionable_flag.isna().sum()==1
    assert read_decisions(tmp_path/'active').decision_data_cutoff_utc.iloc[0]==BundleReader(source).end
    assert pq.read_schema(tmp_path/'active/event-decisions.parquet')==schema('event-decisions')
    assert pd.read_csv(tmp_path/'active/event-decisions.csv').start_utc.iloc[0].endswith('Z')


@pytest.mark.parametrize('case', ['wrong_meter','wrong_direction','too_large','mixed','too_long','not_contained',
                                 'zero_reference','missing_pre','missing_qa','unresolved_qa','pending','disabled',
                                 'right_censored','ambiguous','missing_envelope'])
def test_conservative_keep_cases(tmp_path,case):
    args={}; changes={}
    if case=='wrong_meter': changes['meter_id']='E3020'
    if case=='wrong_direction': args['direction']=-1
    if case=='too_large': args['actuals']={9:200,10:200}
    if case=='mixed': args['actuals']={9:110,10:90}
    if case=='too_long': args['intervals']=((9,16),)
    if case=='not_contained': changes['local_start']='06:00'
    if case=='zero_reference': args['actuals']={i:0 for i in range(4,9)}
    if case=='missing_pre': args['missing']=(6,)
    if case=='disabled': changes['enabled']=False
    if case=='right_censored': args['count']=15
    source=make_bundle(tmp_path/'source',**args)
    cfg=policy_for(source,tmp_path,approved=case not in {'pending','missing_envelope'},changes=changes)
    if case=='missing_qa': cfg['qa_report']=None
    if case=='unresolved_qa':
        path=Path(cfg['qa_report']['path']); qa=json.loads(path.read_text());qa['detectors']['detector-A']['status']='review_required'
        write_json(path,qa);cfg['qa_report']['sha256']=file_hash(path)
    if case=='missing_envelope': cfg['rules'][0]['relative_change_upper']=None
    if case=='ambiguous':
        other=copy.deepcopy(cfg['rules'][0]);other['rule_id']='second';other['reference']=None;other['review']['status']='pending'
        cfg['rules'].append(other)
    report=apply_policy(source,cfg,tmp_path/'preview')
    assert report['effective_events']==1 and report['suppressed_events']==0
    assert read_decisions(tmp_path/'preview').would_suppress.iloc[0] is not np.True_
    assert not bool(read_decisions(tmp_path/'preview').would_suppress.fillna(False).iloc[0])
    if case in {'missing_qa','pending','missing_envelope','ambiguous'}:
        with pytest.raises(ValueError): apply_policy(source,cfg,tmp_path/'active',mode='active')


def test_decrease_and_exact_boundaries(tmp_path):
    source=make_bundle(tmp_path/'source',intervals=((5,20),),direction=-1,count=40)
    cfg=policy_for(source,tmp_path,direction='decrease',changes={'max_duration_seconds':900})
    result=apply_policy(source,cfg,tmp_path/'active',mode='active')
    assert result['suppressed_events']==1  # Exactly 05:55 through 06:10 local.


@pytest.mark.parametrize('date,expected', [('2026-03-07','14:00'),('2026-03-09','13:00'),
                                         ('2026-10-31','13:00'),('2026-11-02','14:00')])
def test_dst_wall_clock_is_stable(tmp_path,date,expected):
    source=make_bundle(tmp_path/'source');rule=policy_for(source,tmp_path)['rules'][0]
    rule.update(local_start='06:00',local_end='06:10')
    start=UTC(f'{date} 12:00');end=UTC(f'{date} 16:00')
    windows,warnings=occurrences(rule,start,end,'America/Los_Angeles')
    assert not warnings and windows[0][0].strftime('%H:%M')==expected


@pytest.mark.parametrize('date,start,end', [('2026-03-08','02:00','02:30'),('2026-11-01','01:00','01:30')])
def test_dst_invalid_boundary_is_not_invented(tmp_path,date,start,end):
    source=make_bundle(tmp_path/'source');rule=policy_for(source,tmp_path)['rules'][0]
    rule.update(local_start=start,local_end=end)
    windows,warnings=occurrences(rule,UTC(f'{date} 07:00'),UTC(f'{date} 13:00'),'America/Los_Angeles')
    assert not windows and 'invalid_dst_occurrence' in warnings


def test_overnight_uses_start_day(tmp_path):
    source=make_bundle(tmp_path/'source');rule=policy_for(source,tmp_path)['rules'][0]
    rule.update(local_start='23:55',local_end='00:10',weekdays=['mon'])
    windows,_=occurrences(rule,UTC('2026-03-03 08:00'),UTC('2026-03-03 08:05'),'America/Los_Angeles')
    assert windows==[(UTC('2026-03-03 07:55'),UTC('2026-03-03 08:10'))]


def test_deterministic_identity_and_mutation_checks(tmp_path):
    source=make_bundle(tmp_path/'source');cfg=policy_for(source,tmp_path)
    a=apply_policy(source,cfg,tmp_path/'a');b=apply_policy(source,cfg,tmp_path/'b')
    assert a['policy_application_id']==b['policy_application_id']
    pd.testing.assert_frame_equal(read_decisions(tmp_path/'a'),read_decisions(tmp_path/'b'))
    moved=copy.deepcopy(cfg)
    for field in ['reference']:
        original=Path(moved['rules'][0][field]['path']);copy_path=tmp_path/'moved-reference.json';copy_path.write_bytes(original.read_bytes())
        moved['rules'][0][field]['path']=str(copy_path)
    c=apply_policy(source,moved,tmp_path/'c')
    assert c['policy_id']==a['policy_id'] and c['policy_application_id']==a['policy_application_id']
    with pytest.raises(FileExistsError): apply_policy(source,cfg,tmp_path/'a')
    changed=copy.deepcopy(cfg);changed['rules'][0]['enabled']=False
    assert apply_policy(source,changed,tmp_path/'changed')['policy_id']!=a['policy_id']
    (tmp_path/'a/event-decisions.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'): verify_policy_bundle(tmp_path/'a',source)
    bad=copy.deepcopy(cfg);bad['qa_report']['sha256']='0'*64
    with pytest.raises(ValueError,match='checksum'): apply_policy(source,bad,tmp_path/'bad')
    qa_path=Path(cfg['qa_report']['path']);qa=json.loads(qa_path.read_text());qa['source_run_id']='another-run'
    write_json(qa_path,qa);cfg['qa_report']['sha256']=file_hash(qa_path)
    with pytest.raises(ValueError,match='identity'):apply_policy(source,cfg,tmp_path/'wrong-qa')


def test_suppressed_truth_is_still_a_miss(tmp_path):
    source=make_bundle(tmp_path/'source');cfg=policy_for(source,tmp_path)
    t=UTC('2026-03-02 13:50')
    annotations=[]
    for a,b,label in [(0,9,'reviewed_normal'),(9,11,'anomaly'),(11,40,'reviewed_normal')]:
        annotations.append({"meter_id": 'E2451',"signal": 'total_active_power',"start": (t+pd.Timedelta(minutes=a)).isoformat(),
                                "end": (t+pd.Timedelta(minutes=b)).isoformat(),"label": label,"label_source": 'artificial',
                                "review_status": 'approved',"review_block": 'full'})
    path=tmp_path/'annotations.json';write_json(path,annotations)
    apply_policy(source,cfg,tmp_path/'active',mode='active',annotations=path)
    result=json.loads((tmp_path/'active/policy-evaluation.json').read_text())['layers']
    raw=result['raw_detector']['results'][0]
    before=result['retained_events_before_policy']['results'][0];after=result['retained_events_after_policy']['results'][0]
    assert raw['point_counts_reviewed_only']['fp']==1 and before['point_counts_reviewed_only']['fp']==0
    assert before['event_recall']==1 and after['event_recall']==0
    assert before['eligible_seconds']==after['eligible_seconds']==2400
    assert before['evaluated_true_events']==after['evaluated_true_events']==1
    assert after['point_counts_reviewed_only']['fn']==2


def test_empty_events_and_unavailable_hours(tmp_path):
    source=make_bundle(tmp_path/'source',count=80,intervals=(),missing=tuple(range(10,80)))
    cfg=policy_for(source,tmp_path)
    result=apply_policy(source,cfg,tmp_path/'overlay')
    assert result['raw_events']==result['effective_events']==0
    assert read_decisions(tmp_path/'overlay').empty
    hours=pd.read_parquet(tmp_path/'overlay/hourly-policy-summary.parquet')
    assert hours.loc[hours.scored_minutes.eq(0),'has_actionable_event_overlap'].isna().all()
    assert pq.read_schema(tmp_path/'overlay/event-decisions.parquet')==schema('event-decisions')


@pytest.mark.parametrize('case',['typo','timezone','duplicate','sign','duration','weekday','date','ref_field'])
def test_strict_policy_validation(tmp_path,case):
    source=make_bundle(tmp_path/'source');cfg=policy_for(source,tmp_path)
    if case=='typo':cfg['timezonee']='UTC'
    elif case=='timezone':cfg['timezone']='Nowhere/Unknown'
    elif case=='duplicate':cfg['rules'].append(copy.deepcopy(cfg['rules'][0]))
    elif case=='sign':cfg['rules'][0]['relative_change_lower']=-.1
    elif case=='duration':cfg['rules'][0]['max_duration_seconds']=61
    elif case=='weekday':cfg['rules'][0]['weekdays']=['monday']
    elif case=='date':cfg['rules'][0]['effective_start']='2026-01-01'
    elif case=='ref_field':cfg['qa_report']['typo']=True
    with pytest.raises(ValueError):validate_policy(cfg)


def test_closed_config_and_cli_default_preview(tmp_path,capsys):
    source=make_bundle(tmp_path/'source');cfg=policy_for(source,tmp_path,approved=False)
    path=tmp_path/'policy.yaml';path.write_text(yaml.safe_dump(cfg))
    assert load_config(path)==cfg
    main(['apply-derconnect-policy',str(source),'--config',str(path),'--out',str(tmp_path/'overlay')])
    assert json.loads(capsys.readouterr().out)['mode']=='preview'
    machine=Path(__file__).resolve().parents[1]/'schemas/derconnect-alert-policy/v1/contract.json'
    assert json.loads(machine.read_text())==contract()


def test_merged_gap_excursion_is_never_hidden(tmp_path):
    source=make_bundle(tmp_path/'source',intervals=((9,18),),actuals={13:200})
    manifest=json.loads((source/'run-manifest.json').read_text())
    name=manifest['prediction_parts'][0];f=pd.read_parquet(source/name)
    f.loc[11:15,'anomaly_flag']=0
    write_table(source/name,f,'predictions');manifest['outputs'][name]=file_hash(source/name)
    write_json(source/'run-manifest.json',manifest)
    cfg=policy_for(source,tmp_path,changes={'max_duration_seconds':900,'local_end':'06:15'})
    result=apply_policy(source,cfg,tmp_path/'overlay',mode='active')
    assert result['suppressed_events']==0
    assert 'change_outside_reviewed_envelope' in result['reason_counts']


def test_code_and_reference_changes_affect_identity(tmp_path,monkeypatch):
    import sarsim_fm.anomaly.alert_policy as module
    source=make_bundle(tmp_path/'source');cfg=policy_for(source,tmp_path)
    first=apply_policy(source,cfg,tmp_path/'first')
    original=module.environment
    def changed():
        value=original();value['source_sha256']='changed-code';return value
    monkeypatch.setattr(module,'environment',changed)
    second=apply_policy(source,cfg,tmp_path/'second')
    assert first['policy_application_id']!=second['policy_application_id']
    path=Path(cfg['rules'][0]['reference']['path']);ref=json.loads(path.read_text())
    ref['relative_change_upper']=.9;write_json(path,ref)
    cfg['rules'][0]['reference']['sha256']=file_hash(path)
    with pytest.raises(ValueError,match='reference evidence'):
        apply_policy(source,cfg,tmp_path/'wrong-reference')


def test_interrupted_overlay_is_not_complete(tmp_path,monkeypatch):
    import sarsim_fm.anomaly.alert_policy as module
    source=make_bundle(tmp_path/'source');cfg=policy_for(source,tmp_path)
    original=module.write_table
    count=0
    def interrupted(*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:raise RuntimeError('interruption')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'write_table',interrupted)
    with pytest.raises(RuntimeError):apply_policy(source,cfg,tmp_path/'partial')
    assert not (tmp_path/'partial/policy-manifest.json').exists()
    with pytest.raises(FileNotFoundError):verify_policy_bundle(tmp_path/'partial',source)


def test_no_labels_remain_unknown_after_suppression(tmp_path):
    source=make_bundle(tmp_path/'source');cfg=policy_for(source,tmp_path)
    labels=tmp_path/'unknown.json';write_json(labels,[])
    apply_policy(source,cfg,tmp_path/'overlay',mode='active',annotations=labels)
    result=json.loads((tmp_path/'overlay/policy-evaluation.json').read_text())
    for layer in result['layers'].values():
        row=layer['results'][0]
        assert row['event_recall'] is None and row['point_fpr'] is None
        assert row['point_counts_reviewed_only']=={'tp':0,'fp':0,'tn':0,'fn':0}
        assert row['eligible_seconds']==2400
