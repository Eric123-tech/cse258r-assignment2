"""Small hand-computable checks for metrics, similarity and temporal isolation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy import sparse
import pipeline as p

# Use >=10 artists to satisfy the real top-10 context invariant.
h=pd.DataFrame([dict(user_index=u,artist_index=a,timestamp=pd.Timestamp('2009-01-01')) for u in range(3) for a in range(12) if a%3==u or a==0])
c=p.make_context(h,'2009-02-01',min_listeners=1,min_events=1)
future=pd.DataFrame({'user_index':[0,0], 'artist_index':[0,999]})
scores=np.tile(np.arange(12,0,-1),(3,1))
s,users=p.evaluate(scores,c,future,k=2)
assert s['recall']==.5 and s['precision']==.5
assert np.isclose(s['ndcg'],1/(1+1/np.log2(3)))
assert s['target_coverage']==.5
assert users.iloc[0].novel_recall==0
# Unit self-neighbor count=0 would zero all predictions.
assert np.allclose(p.cf_scores(c,'cosine',neighbors=0),0)
assert np.allclose(p.cf_scores(c,'jaccard',neighbors=0),0)
r=p.recent_matrix(c,half_life=31)
assert np.allclose(r.data,.5*c['counts'].data)
assert 999 not in c['artists']
assert set(c['users'])=={0,1,2}
assert p.paired_bootstrap(users,users)['mean_difference']==0
st=p.summary_table(pd.DataFrame([s],index=['m']),{'m':users},'m')
assert st.loc['m','diff_vs_m']==0 and st.loc['m','ndcg_ci_low']<=st.loc['m','ndcg']<=st.loc['m','ndcg_ci_high']
# Global chronological boundaries are disjoint and retain all valid events.
time_events=pd.DataFrame({'timestamp':pd.date_range('2009-01-01',periods=100,freq='D')})
a,b,t,bounds=p.temporal_split(time_events)
assert len(a)+len(b)+len(t)==len(time_events)
assert a.timestamp.max()<b.timestamp.min() and b.timestamp.max()<t.timestamp.min()
for model in p.candidates():
    if model['kind']=='logistic':continue
    scores=p.model_scores(c,model)
    assert scores.shape==c['counts'].shape and np.isfinite(scores).all(),model
assert np.allclose(p.model_scores(c,dict(kind='popular'))[1],c['popularity'])
# Learned ranker: trains on [cutoff-28d, cutoff) labels from history only; replays of artists<6 give positives.
h3=pd.concat([h,h[h.artist_index<6].assign(timestamp=pd.Timestamp('2009-01-20'))],ignore_index=True)
c3=p.make_context(h3,'2009-02-01',min_listeners=1,min_events=1)
_,_,info=p.fit_ranker(c3)
assert 0<info['positive_rate']<1
s3=p.model_scores(c3,dict(kind='logistic',C=1.0))
assert s3.shape==c3['counts'].shape and np.isfinite(s3).all()
assert (s3[c3['counts'].toarray()>0].min()>s3.min())  # every heard artist outranks non-candidates
# Representation transformations must change vote strength, not binary neighbors.
h2 = pd.concat([h,h[(h.user_index==0) & (h.artist_index==0)]],ignore_index=True)
c2=p.make_context(h2,'2009-02-01',min_listeners=1,min_events=1)
row=p.preference_matrix(c2,'count').toarray()[0]
binary=p.preference_matrix(c2,'binary').toarray()[0]
log=p.preference_matrix(c2,'log1p').toarray()[0]
assert row[0]>log[0]>binary[0]
for rep in ['binary','count','log1p']:
    x=p.preference_matrix(c2,rep)
    assert np.allclose(np.asarray(x.sum(axis=1)).ravel(),1)
    assert np.isfinite(p.cf_scores(c2,'jaccard',2,rep)).all()
assert users.iloc[0].repeat_recall==1
# Listening-time proxies: gaps to next play within history; final play has no gap and keeps 1.
g=pd.DataFrame({'user_index':0,'track_index':range(4),'timestamp':pd.Timestamp('2009-01-01')+pd.to_timedelta([0,100,110,800],unit='s')})
assert np.allclose(p.play_weights(g,'gap'),[100/600,10/600,1,1])
assert np.allclose(p.play_weights(g,'no_skip'),[1,0,1,1])
assert np.allclose(p.play_weights(g,'relative'),[1,10/395,1,1])
assert np.allclose(p.recent_matrix(c,31,'count').data,r.data)
# Reject leaked boundary events, invalid weights and unmatched paired populations.
for operation in [lambda:p.make_context(h,'2009-01-01',min_listeners=1,min_events=1),
                  lambda:p.recent_matrix(c,0),
                  lambda:p.evaluate(np.zeros((1,1)),c,future),
                  lambda:p.paired_bootstrap(users,users.iloc[0:0])]:
    try:operation()
    except ValueError:pass
    else:raise AssertionError('Expected invalid protocol input to be rejected')
print('PASS: ranking/OOV/repeat metrics, temporal boundaries, weights, feature representations, paired populations and model shapes.')
