import json

import pandas as pd
import pytest

from sarsim_fm.anomaly.review_annotations import normalize_reviews


def inputs(tmp_path):
    mapping={'status':'approved','reviewer':'synthetic reviewer','reviewed_at':'2026-01-01T00:00:00Z',
             'label_source':'artificial-review','categories':{'expected_operation':'reviewed_normal',
              'measurement_issue':'unknown','potential_operational_anomaly':'anomaly','uncertain':'unknown'}}
    rows=[]
    for i,category in enumerate(mapping['categories']):
        start=pd.Timestamp('2026-03-01T00:00:00Z')+pd.Timedelta(hours=i)
        rows.append({'meter_id':'E2451','signal':'total_active_power','start':start.isoformat(),
                     'end':(start+pd.Timedelta(minutes=10)).isoformat(),'category':category,
                     'review_status':'approved','reviewer':'synthetic reviewer'})
    path=tmp_path/'mapping.json';path.write_text(json.dumps(mapping))
    reviews=tmp_path/'reviews.csv';pd.DataFrame(rows).to_csv(reviews,index=False)
    return reviews,path,rows,mapping


def test_normalization_never_promotes_potential_or_quality_without_confirmation(tmp_path):
    reviews,path,_,_=inputs(tmp_path)
    result=normalize_reviews(reviews,path,tmp_path/'truth.json')
    assert [r['label'] for r in result]==['reviewed_normal','unknown','unknown','unknown']
    assert (tmp_path/'truth.provenance.json').exists()
    with pytest.raises(FileExistsError):normalize_reviews(reviews,path,tmp_path/'truth.json')


def test_confirmed_truth_and_pending_review(tmp_path):
    reviews,path,rows,_=inputs(tmp_path)
    rows[2]['truth_confirmed']=True
    rows[0]['review_status']='pending'
    pd.DataFrame(rows).to_csv(reviews,index=False)
    result=normalize_reviews(reviews,path,tmp_path/'truth.json')
    assert result[0]['label']=='unknown' and result[2]['label']=='anomaly'


def test_unapproved_mapping_and_overlap_fail(tmp_path):
    reviews,path,rows,mapping=inputs(tmp_path)
    mapping['status']='pending';path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError,match='approved'):normalize_reviews(reviews,path,tmp_path/'truth.json')
    mapping['status']='approved';path.write_text(json.dumps(mapping))
    pd.DataFrame(rows+[rows[0]]).to_csv(reviews,index=False)
    with pytest.raises(ValueError,match='overlap'):normalize_reviews(reviews,path,tmp_path/'truth.json')
    assert not (tmp_path/'truth.json').exists()


def test_isolated_flag_review_sampling_is_reproducible(tmp_path):
    from sarsim_fm.anomaly.derconnect_review import review_bundle
    from test_derconnect_alert_policy import make_bundle
    source=make_bundle(tmp_path/'source')
    a=review_bundle(source,tmp_path/'a',isolated_count=10)
    b=review_bundle(source,tmp_path/'b',isolated_count=10)
    assert a==b
    assert sum(r['selection_reason']=='isolated_raw_flag' for r in a)==1
    assert all(r['review_status']=='pending' and r['category']=='uncertain' for r in a)
