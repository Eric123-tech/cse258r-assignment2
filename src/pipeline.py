"""Auditable temporal artist recommendation; no sampled evaluation negatives."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
SEED = 258

def load_events():
    (ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
    parts, audit = [], []
    for split in ['train', 'valid', 'test']:
        p = ROOT / f'data/raw/{split}.gz.parquet'
        f = pq.ParquetFile(p)
        audit.append({'source_split': split, 'rows': f.metadata.num_rows})
        for batch in f.iter_batches(columns=['user_index','artist_index','track_index','timestamp'], batch_size=500000):
            x = batch.to_pandas()
            for c in ['user_index','artist_index','track_index']:
                x[c] = x[c].astype('int32')
            parts.append(x)
    events = pd.concat(parts, ignore_index=True)
    del parts
    before = len(events)
    events = events.dropna().drop_duplicates(['user_index','artist_index','track_index','timestamp'])
    # Conservative exclusive May 5 cutoff from the original source description.
    # Keep raw files intact; write excluded records for audit.
    in_period = events.timestamp < pd.Timestamp('2009-05-05')
    excluded = events.loc[~in_period]
    excluded.to_parquet(ROOT / 'data/processed/excluded_timestamps.parquet', index=False)
    audit.append({'source_split':'outside documented period (>=2009-05-05)', 'rows':len(excluded)})
    events = events.loc[in_period].sort_values('timestamp').reset_index(drop=True)
    audit.append({'source_split':'after null/exact-duplicate removal', 'rows':len(events), 'removed':before-len(events)-len(excluded)})
    return events, pd.DataFrame(audit)

def temporal_split(events, days=28):
    end = events.timestamp.max().floor('D') + pd.Timedelta(days=1)
    test_start = end - pd.Timedelta(days=days)
    valid_start = test_start - pd.Timedelta(days=days)
    train = events[events.timestamp < valid_start]
    valid = events[(events.timestamp >= valid_start) & (events.timestamp < test_start)]
    test = events[(events.timestamp >= test_start) & (events.timestamp < end)]
    assert len(train) and len(valid) and len(test)
    assert train.timestamp.max() < valid_start <= valid.timestamp.min()
    assert valid.timestamp.max() < test_start <= test.timestamp.min()
    return train, valid, test, dict(valid_start=str(valid_start), test_start=str(test_start), end=str(end), window_days=days)

def make_context(history, cutoff, min_listeners=5, min_events=100):
    """Catalog and warm-user eligibility are determined exclusively from history."""
    counts = history.groupby(['user_index','artist_index']).size().rename('count').reset_index()
    listeners = counts.groupby('artist_index').size()
    artists = np.sort(listeners[listeners >= min_listeners].index.to_numpy())
    user_counts = history.groupby('user_index').size()
    users = np.sort(user_counts.index.to_numpy())
    eligible = set(user_counts[user_counts >= min_events].index)
    ui = pd.Index(users).get_indexer(counts.user_index)
    ai = pd.Index(artists).get_indexer(counts.artist_index)
    ok = ai >= 0
    matrix = sparse.csr_matrix((counts['count'].to_numpy()[ok].astype(float), (ui[ok], ai[ok])), shape=(len(users),len(artists)))
    assert len(artists) >= 10
    binary = matrix.copy(); binary.data[:] = 1
    return dict(history=history, cutoff=pd.Timestamp(cutoff), users=users, artists=artists,
                eligible=eligible, history_sets=history.groupby('user_index').artist_index.agg(set).to_dict(),
                history_sizes=user_counts.to_dict(), counts=matrix, binary=binary, popularity=np.asarray(binary.sum(axis=0)).ravel())

def recent_matrix(ctx, half_life):
    h = ctx['history']
    age = (ctx['cutoff'] - h.timestamp).dt.total_seconds().to_numpy() / 86400
    assert np.min(age) >= 0
    w = np.exp2(-age / half_life)
    # Aggregate before creating sparse matrix to bound memory.
    grouped = pd.DataFrame({'u':h.user_index.to_numpy(), 'a':h.artist_index.to_numpy(), 'w':w}).groupby(['u','a']).w.sum().reset_index()
    ui = pd.Index(ctx['users']).get_indexer(grouped.u)
    ai = pd.Index(ctx['artists']).get_indexer(grouped.a)
    ok = ai >= 0
    return sparse.csr_matrix((grouped.w.to_numpy()[ok], (ui[ok], ai[ok])), shape=ctx['counts'].shape)

def cf_scores(ctx, metric='cosine', neighbors=40):
    b = ctx['binary']
    if metric == 'cosine':
        z = normalize(b, axis=1)
        sim = (z @ z.T).toarray()
    elif metric == 'jaccard':
        inter = (b @ b.T).toarray()
        n = np.asarray(b.sum(axis=1)).ravel()
        union = n[:,None] + n[None,:] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union>0)
    else:
        raise ValueError(metric)
    np.fill_diagonal(sim, 0)
    # Stable sorting makes tied neighbor selection deterministic.
    order = np.argsort(-sim, axis=1, kind='stable')[:, :neighbors]
    keep = np.zeros_like(sim)
    np.put_along_axis(keep, order, np.take_along_axis(sim, order, axis=1), axis=1)
    preferences = ctx['counts'].copy(); preferences.data = np.log1p(preferences.data)
    preferences = normalize(preferences, norm='l1', axis=1)
    return (sparse.csr_matrix(keep) @ preferences).toarray()

def model_scores(ctx, spec):
    kind = spec['kind']
    if kind == 'popular':
        return np.broadcast_to(ctx['popularity'], ctx['counts'].shape)
    if kind == 'personal':
        return ctx['counts'].toarray()
    if kind == 'recent':
        return recent_matrix(ctx,spec['half_life']).toarray()
    if kind == 'cf':
        return cf_scores(ctx, spec['metric'], spec['neighbors'])
    if kind == 'svd':
        x = ctx['counts'].copy(); x.data = np.log1p(x.data)
        x = normalize(x, norm='l2', axis=1)
        model = TruncatedSVD(n_components=min(spec['rank'], min(x.shape)-1), random_state=SEED)
        factors = model.fit_transform(x)
        return factors @ model.components_
    if kind == 'hybrid':
        cf = normalize(cf_scores(ctx,'cosine',40),norm='l1',axis=1)
        recent = normalize(recent_matrix(ctx,spec['half_life']),norm='l1',axis=1).toarray()
        return (1-spec['alpha'])*cf + spec['alpha']*recent
    raise ValueError(kind)

def evaluate(scores, ctx, future, k=10):
    """Recall denominator includes out-of-catalog positives (honest end-to-end recall)."""
    truth = future.groupby('user_index').artist_index.agg(lambda x:set(x))
    catalog = set(ctx['artists'])
    pop = ctx['popularity']
    catalog_order = np.lexsort((ctx['artists'], -pop))
    tail = set(ctx['artists'][catalog_order[len(catalog_order)//5:]])
    rows = []
    for row,u in enumerate(ctx['users']):
        if u not in ctx['eligible'] or u not in truth.index:
            continue
        target = truth.loc[u]
        # Primary score, then training-only popularity, then artist ID.
        idx = np.lexsort((ctx['artists'], -pop, -scores[row]))[:k]
        rec = ctx['artists'][idx]
        hits = np.array([a in target for a in rec],dtype=float)
        ideal = (1/np.log2(np.arange(2,min(k,len(target))+2))).sum()
        hist = ctx['history_sets'][u]
        novel = target - hist
        rows.append(dict(user_index=int(u), recall=hits.sum()/len(target), precision=hits.mean(),
                         ndcg=float((hits/np.log2(np.arange(2,k+2))).sum()/ideal),
                         target_artists=len(target), covered_artists=len(target & catalog),
                         repeat_share=len(target & hist)/len(target),
                         novel_recall=len(set(rec)&novel)/len(novel) if novel else np.nan,
                         tail_recall=len(set(rec)&target&tail)/len(target&tail) if target&tail else np.nan,
                         history_events=int(ctx['history_sizes'][u]), recommendations=rec.tolist()))
    if not rows:
        raise ValueError('No eligible evaluation users; inspect history and windows.')
    per_user = pd.DataFrame(rows)
    summary = per_user[['recall','precision','ndcg','novel_recall','tail_recall']].mean().to_dict()
    summary.update(users=len(rows), target_coverage=per_user.covered_artists.sum()/per_user.target_artists.sum(),
                   recommendation_coverage=len(set(a for r in per_user.recommendations for a in r))/len(catalog))
    return summary, per_user

def candidates():
    """Two course-related baselines plus one project temporal method."""
    specs = [dict(name=f'UserCF {m} k={k}',kind='cf',metric=m,neighbors=k)
             for m in ['cosine','jaccard'] for k in [20,40]]
    specs += [dict(name=f'Recency h={h}d',kind='recent',half_life=h) for h in [7,28,90]]
    return specs

def compare(ctx,future,specs):
    results,details = [],{}
    for spec in specs:
        print('Evaluating:',spec['name'],flush=True)
        summary,users = evaluate(model_scores(ctx,spec),ctx,future)
        results.append(dict(model=spec['name'],**summary)); details[spec['name']] = users
    return pd.DataFrame(results).set_index('model'),details

def paired_bootstrap(a,b,metric='ndcg', repetitions=2000):
    merged = a[['user_index',metric]].merge(b[['user_index',metric]],on='user_index',validate='one_to_one')
    d = (merged[f'{metric}_x']-merged[f'{metric}_y']).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = rng.choice(d,size=(repetitions,len(d)),replace=True).mean(axis=1)
    return dict(mean_difference=float(d.mean()), ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),users=len(d))
