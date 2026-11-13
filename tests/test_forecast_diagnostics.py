"""Saved-output QA and fake pinned reproduction; no foundation inference or network."""
import json

import numpy as np
import pandas as pd
import pytest

from sarsim_fm.anomaly.export import write_json, write_table
from sarsim_fm.anomaly.forecast import rolling_forecast
from sarsim_fm.anomaly.forecast_diagnostics import (
    extract_context,
    inspect_bundle,
    predict_cases,
    reproduce,
    reproduction_scope,
)
from sarsim_fm.anomaly.provenance import file_hash
from test_derconnect_alert_policy import make_bundle


def context_stage(root):
    root.mkdir()
    times=pd.date_range('2026-03-01',periods=3000,freq='min',tz='UTC')
    frame=pd.DataFrame({'timestamp_utc':times,'actual':np.arange(len(times),dtype=float)+100,'usable':True})
    names=[]
    for day,f in frame.groupby(frame.timestamp_utc.dt.strftime('%Y-%m-%d')):
        name=f'prepared/E2451/{day}.parquet';path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        f.to_parquet(path,index=False);names.append(name)
    write_json(root/'stage-manifest.json',{'complete':True,'preparation_id':'prepared-fixture',
               'prepared_parts':names,'prepared_hashes':{n:file_hash(root/n) for n in names}})
    return root


def test_context_excludes_future_and_detects_mutation(tmp_path):
    root=context_stage(tmp_path/'stage');target='2026-03-02T01:00:00Z'
    x,meta=extract_context(root,'E2451',target,1440)
    assert len(x)==1440 and meta['end_utc']==pd.Timestamp(target).isoformat()
    assert x[-1]==1599
    path=root/'prepared/E2451/2026-03-02.parquet'
    f=pd.read_parquet(path);f.loc[f.timestamp_utc>=pd.Timestamp(target),'actual']=999999
    f.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='hash mismatch'):extract_context(root,'E2451',target,1440)
    manifest=json.loads((root/'stage-manifest.json').read_text())
    manifest['prepared_hashes']['prepared/E2451/2026-03-02.parquet']=file_hash(path)
    write_json(root/'stage-manifest.json',manifest)
    other,_=extract_context(root,'E2451',target,1440)
    np.testing.assert_equal(x,other)
    f.loc[f.timestamp_utc==pd.Timestamp(target)-pd.Timedelta(minutes=2),'usable']=False
    f.to_parquet(path,index=False);manifest['prepared_hashes']['prepared/E2451/2026-03-02.parquet']=file_hash(path)
    write_json(root/'stage-manifest.json',manifest)
    with pytest.raises(ValueError,match='uninterrupted'):extract_context(root,'E2451',target,1440)
    with pytest.raises(ValueError,match='insufficient'):extract_context(root,'E2451','2026-03-01T00:01:00Z',256)


class Fake:
    def predict_quantiles(self,contexts,horizon,levels,batch_size,season_lengths):
        return np.array([[[x[-1]+2,x[-1]-2,x[-1],x[-1]+1,x[-1]-1]] for x in contexts])


def test_raw_quantiles_match_production_rearrangement():
    x=np.arange(257.)
    raw,exported,score=predict_cases(Fake(),[x[:256]],[x[256]],1)
    production=rolling_forecast(Fake(),'case',x,256,256,[.01,.1,.5,.9,.99],1)
    assert (np.diff(raw,axis=1)<0).any()
    np.testing.assert_equal(exported,production.quantiles)
    assert score[0]==0


def test_inspection_captures_instability_without_loading(tmp_path,monkeypatch):
    source=make_bundle(tmp_path/'source',missing=(20,))
    manifest=json.loads((source/'run-manifest.json').read_text())
    name=manifest['prediction_parts'][0];f=pd.read_parquet(source/name)
    f.loc[9,['q01','q50','anomaly_score']]=[-300105088.,-70813.765625,0.]
    write_table(source/name,f,'predictions')
    manifest['outputs'][name]=file_hash(source/name);write_json(source/'run-manifest.json',manifest)
    from sarsim_fm import models
    monkeypatch.setattr(models,'load_derconnect_forecaster',lambda *a:pytest.fail('offline inspection loaded a model'))
    targets=[f.timestamp_utc.iloc[9].isoformat(),f.timestamp_utc.iloc[20].isoformat()]
    report=inspect_bundle(source,tmp_path/'qa',targets)
    assert report['detectors']['detector-A']['status']=='review_required'
    assert report['detectors']['detector-A']['negative_median_positive_actual']==1
    assert report['cases'][0]['raw_unrearranged_quantiles'] is None
    assert report['cases'][0]['score']==0 and report['cases'][1]['actual'] is None
    assert (tmp_path/'qa/cases.svg').exists()
    scope=reproduction_scope(source,targets)
    assert scope['maximum_target_predictions']==8 and not scope['weights_loaded']
    with pytest.raises(FileExistsError):inspect_bundle(source,tmp_path/'qa',targets)


def test_degenerate_baseline_has_no_probabilistic_qa_requirement(tmp_path):
    source=make_bundle(tmp_path/'source')
    manifest=json.loads((source/'run-manifest.json').read_text());manifest['quantile_kind']='degenerate_point'
    write_json(source/'run-manifest.json',manifest)
    result=inspect_bundle(source,tmp_path/'qa',[])
    assert result['detectors']['detector-A']['status']=='pass'


def test_fake_reproduction_is_bounded_and_pinned(tmp_path):
    source=make_bundle(tmp_path/'source',count=30,start='2026-03-02 01:00')
    stage=context_stage(tmp_path/'stage')
    targets=['2026-03-02T01:05:00Z','2026-03-02T01:06:00Z']
    report=reproduce(source,stage,tmp_path/'repro',forecaster=Fake(),targets=targets)
    assert report['target_predictions']==8
    assert all(r['raw_crossing'] for r in report['records'])
    assert all(r['batch_difference_max_abs']==0 for r in report['records'])
    assert report['model_parameter_dtype']=='test_double'
    for r in report['records']:
        assert pd.Timestamp(r['context']['end_utc'])==pd.Timestamp(r['target'])
    assert (tmp_path/'repro/diagnostic-manifest.json').exists()


def test_prepared_nrp_handoff_is_only_files_and_caps_predictions(tmp_path):
    import importlib.util
    import tarfile
    from pathlib import Path

    import yaml
    script=Path(__file__).resolve().parents[1]/'scripts/derconnect/prepare_nrp_diagnostics.py'
    spec=importlib.util.spec_from_file_location('diagnostic_handoff',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=make_bundle(tmp_path/'source');stage=context_stage(tmp_path/'stage')
    manifest=json.loads((source/'run-manifest.json').read_text());manifest['preparation_id']='prepared-fixture'
    write_json(source/'run-manifest.json',manifest)
    out=tmp_path/'ali-diagnostics'
    result=module.prepare(source,stage,out,'/data/ali/diagnostics')
    assert result['maximum_target_predictions']==40
    job=yaml.safe_load((out/'diagnostic-job.yaml').read_text())
    command=job['spec']['template']['spec']['containers'][0]['command'][-1]
    assert 'reproduce' in command and 'sfm train' not in command and 'sleep' not in command
    assert json.loads((out/'handoff-manifest.json').read_text())['submitted'] is False
    with tarfile.open(out/'inputs.tar.gz') as archive:
        assert not any('/raw/' in n or n.endswith('.csv') and 'prepared/' in n for n in archive.getnames())
    with pytest.raises(ValueError,match='1..10'):
        reproduction_scope(source,[f'2026-03-02T01:{i:02d}:00Z' for i in range(11)])


def test_nonfinite_reproduction_is_saved_as_failure_not_normal(tmp_path):
    class BadFake:
        def predict_quantiles(self,contexts,horizon,levels,batch_size,season_lengths):
            return np.full((len(contexts),1,5),np.nan)
    source=make_bundle(tmp_path/'source',count=30,start='2026-03-02 01:00')
    stage=context_stage(tmp_path/'stage')
    report=reproduce(source,stage,tmp_path/'repro',forecaster=BadFake(),targets=['2026-03-02T01:05:00Z'])
    assert report['target_predictions']==report['failed_target_predictions']==4
    assert report['successful_target_predictions']==0
    assert all(r['production_would_reject'] and r['score'] is None for r in report['records'])
    assert all(r['raw_quantiles']==[None]*5 for r in report['records'])
