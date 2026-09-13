"""T016R: tolerate only descriptive per-episode JS float64 replay roundoff."""
import math


def compare_historical_episode_rows(replay,historical,replay_headers,historical_headers):
    assert replay_headers==historical_headers
    assert len(replay)==len(historical)
    deltas=[]
    for index,(new,old) in enumerate(zip(replay,historical)):
        assert list(new)==list(old)==replay_headers
        assert all(new[k]==old[k] for k in replay_headers if k!='JS')
        a,b=float(new['JS']),float(old['JS'])
        assert math.isfinite(a) and math.isfinite(b)
        delta=abs(a-b)
        assert delta<=1e-12
        if delta:
            deltas.append(dict(row_index=index,identifying_fields={k:old[k] for k in ('client','bank','target','estimator')},historical_value=b,replay_value=a,absolute_delta=delta))
    return dict(field='JS',atol=1e-12,rtol=0,rows_checked=len(replay),nonzero_delta_count=len(deltas),max_absolute_delta=max((r['absolute_delta'] for r in deltas),default=0.),deltas=deltas,all_non_JS_fields_exact=True,schema_row_count_order_exact=True)


def compare_historical_table(name,replay,historical,replay_headers,historical_headers):
    assert replay_headers==historical_headers
    if name=='mixture_quality_episodes.csv':
        return compare_historical_episode_rows(replay,historical,replay_headers,historical_headers)
    assert replay==historical
    return dict(file=name,exact=True,rows=len(replay))
