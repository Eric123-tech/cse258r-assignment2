"""Auditable temporal artist recommendation; no sampled evaluation negatives."""
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LogisticRegression

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
    events = events.dropna()
    missing = before - len(events)
    nonnull = len(events)
    events = events.drop_duplicates(['user_index','artist_index','track_index','timestamp'])
    duplicates = nonnull - len(events)
    # Conservative exclusive cutoff from the creator's collection description.
    # Later records are excluded by provenance, not individually proven corrupt.
    in_period = events.timestamp < pd.Timestamp('2009-05-05')
    excluded = events.loc[~in_period]
    excluded.to_parquet(ROOT / 'data/processed/excluded_timestamps.parquet', index=False)
    events = events.loc[in_period].sort_values('timestamp').reset_index(drop=True)
    audit += [dict(source_split='required-field nulls removed', rows=missing),
              dict(source_split='exact duplicate events removed', rows=duplicates),
              dict(source_split='outside documented period removed', rows=len(excluded)),
              dict(source_split='retained after all preprocessing', rows=len(events))]
    assert before == missing + duplicates + len(excluded) + len(events)
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
    if history.empty or history.timestamp.max() >= pd.Timestamp(cutoff):
        raise ValueError('History must be nonempty and strictly before the prediction cutoff.')
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
                thresholds=dict(min_listeners=min_listeners, min_events=min_events),
                eligible=eligible, history_sets=history.groupby('user_index').artist_index.agg(set).to_dict(),
                history_sizes=user_counts.to_dict(), counts=matrix, binary=binary, popularity=np.asarray(binary.sum(axis=0)).ravel())

def play_weights(h, kind='count', cap=600, skip=30):
    """Listening-time proxies from the gap to the same user's next historical play.
    Scrobble timestamps mark track starts, so a gap ~ play time plus any pause. The last
    play before the cutoff has no observed gap (the next play is unseen) and keeps weight 1."""
    if kind == 'count':
        return np.ones(len(h))
    # History is globally time-sorted, so within-user shift is chronological.
    gap = (h.groupby('user_index').timestamp.shift(-1) - h.timestamp).dt.total_seconds()
    if kind == 'gap':
        return gap.clip(upper=cap).div(cap).fillna(1).to_numpy()
    if kind == 'no_skip':
        return (gap.isna() | (gap >= skip)).astype(float).to_numpy()
    if kind == 'relative':
        # Typical track length: median plausible gap per track, else the global median.
        plausible = gap.where(gap.between(skip, 1200))
        typical = plausible.groupby(h.track_index).transform('median').fillna(plausible.median())
        return (gap / typical).clip(upper=1).fillna(1).to_numpy()
    raise ValueError(f'Unknown play weight: {kind}')

def recent_matrix(ctx, half_life, play_weight='count'):
    if half_life <= 0:
        raise ValueError('half_life must be positive.')
    h = ctx['history']
    age = (ctx['cutoff'] - h.timestamp).dt.total_seconds().to_numpy() / 86400
    assert np.min(age) >= 0
    w = np.exp2(-age / half_life) * play_weights(h, play_weight)
    # Aggregate before creating sparse matrix to bound memory.
    grouped = pd.DataFrame({'u':h.user_index.to_numpy(), 'a':h.artist_index.to_numpy(), 'w':w}).groupby(['u','a']).w.sum().reset_index()
    ui = pd.Index(ctx['users']).get_indexer(grouped.u)
    ai = pd.Index(ctx['artists']).get_indexer(grouped.a)
    ok = ai >= 0
    return sparse.csr_matrix((grouped.w.to_numpy()[ok], (ui[ok], ai[ok])), shape=ctx['counts'].shape)

def preference_matrix(ctx, representation='log1p'):
    """Neighbor voting features, holding binary user similarity fixed."""
    x = ctx['counts'].copy()
    if representation == 'binary':
        x.data[:] = 1
    elif representation == 'log1p':
        x.data = np.log1p(x.data)
    elif representation != 'count':
        raise ValueError(f'Unknown preference representation: {representation}')
    return normalize(x, norm='l1', axis=1)


def cf_scores(ctx, metric='cosine', neighbors=40, representation='log1p'):
    """Course neighborhood baseline; representations change voting, not neighbors."""
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
    preferences = preference_matrix(ctx, representation)
    return (sparse.csr_matrix(keep) @ preferences).toarray()

NEVER = 3650  # days-since-play placeholder for CF-only candidates the user never played

def pair_features(ctx, extra=20):
    """One row per (user, candidate artist): every historical artist plus top-`extra` unheard CF artists."""
    h = ctx['history']
    h = h[h.artist_index.isin(ctx['artists'])]
    age = (ctx['cutoff'] - h.timestamp).dt.total_seconds().to_numpy() / 86400
    x = pd.DataFrame({'u':h.user_index.to_numpy(), 'a':h.artist_index.to_numpy(), 'age':age, 'day':np.floor(age)})
    for hl in [3, 7, 28, 90, 365]:
        x[f'd{hl}'] = np.exp2(-age / hl)
    x['p28'] = age < 28
    x['day28'] = np.where(age < 28, x.day, np.nan)
    f = x.groupby(['u','a']).agg(count=('age','size'), last=('age','min'), first=('age','max'),
                                 d3=('d3','sum'), d7=('d7','sum'), d28=('d28','sum'), d90=('d90','sum'), d365=('d365','sum'),
                                 p28=('p28','sum'), days28=('day28','nunique'), days_all=('day','nunique')).reset_index()
    cf = cf_scores(ctx, 'jaccard', 20)
    users, artists = pd.Index(ctx['users']), pd.Index(ctx['artists'])
    seen = set(zip(f.u, f.a))
    order = np.argsort(-cf, axis=1, kind='stable')
    rows = []
    for r, u in enumerate(ctx['users']):
        new = [a for a in artists[order[r, :len(ctx['history_sets'].get(u, ())) + extra]] if (u, a) not in seen][:extra]
        rows += [(u, a) for a in new]
    f = pd.concat([f, pd.DataFrame(rows, columns=['u','a'])], ignore_index=True)
    f[['last','first']] = f[['last','first']].fillna(NEVER)
    f = f.fillna(0)
    user = x.groupby('u').agg(u_total=('age','size'), u_p28=('p28','sum'), u_last=('age','min'))
    f = f.join(user, on='u').join(f[f['count'] > 0].groupby('u').size().rename('u_artists'), on='u').fillna({'u_last':NEVER}).fillna(0)
    f['share28'] = f.p28 / f.u_p28.clip(lower=1)
    f['share_all'] = f['count'] / f.u_total.clip(lower=1)
    f['cf'] = cf[users.get_indexer(f.u), artists.get_indexer(f.a)]
    f['pop'] = ctx['popularity'][artists.get_indexer(f.a)]
    f['pop28'] = f.a.map(x[x.age < 28].groupby('a').u.nunique()).fillna(0)
    f['heard'] = (f['count'] > 0).astype(float)
    return f

RANKER_FEATURES = ['count','last','first','d3','d7','d28','d90','d365','p28','days28','days_all',
                   'u_total','u_p28','u_last','u_artists','share28','share_all','cf','pop','pop28','heard']

def feature_matrix(f):
    z = f[RANKER_FEATURES].astype(float)
    logged = [c for c in RANKER_FEATURES if c not in ('share28','share_all','heard')]
    z[logged] = np.log1p(z[logged].clip(lower=0))
    return z.to_numpy()

def fit_ranker(ctx, C=1.0, window=28):
    """Learn from the window just before ctx's cutoff: features from history before cutoff-window,
    labels = artists heard in [cutoff-window, cutoff). Uses only ctx['history'], so no future leakage."""
    split = ctx['cutoff'] - pd.Timedelta(days=window)
    h = ctx['history']
    inner = make_context(h[h.timestamp < split], split, **ctx['thresholds'])
    future = h[h.timestamp >= split]
    f = pair_features(inner)
    f = f[f.u.isin(inner['eligible'] & set(future.user_index))]
    y = f.merge(future[['user_index','artist_index']].drop_duplicates(), how='left', left_on=['u','a'],
                right_on=['user_index','artist_index'], indicator=True)['_merge'].eq('both').to_numpy()
    if y.all() or not y.any():
        raise ValueError('Ranker training window needs both positive and negative pairs.')
    scaler = StandardScaler().fit(feature_matrix(f))
    model = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(feature_matrix(f)), y)
    return scaler, model, dict(pairs=len(f), positive_rate=float(y.mean()))

def ranker_scores(ctx, C=1.0):
    scaler, model, _ = fit_ranker(ctx, C)
    f = pair_features(ctx)
    s = model.decision_function(scaler.transform(feature_matrix(f)))
    # Non-candidates rank below every candidate; evaluate() then breaks ties by popularity.
    out = np.full(ctx['counts'].shape, s.min() - 1.0)
    out[pd.Index(ctx['users']).get_indexer(f.u), pd.Index(ctx['artists']).get_indexer(f.a)] = s
    return out

def model_scores(ctx, spec):
    """Trivial baselines, the two course baselines, the temporal method and the learned ranker."""
    if spec['kind'] == 'popular':
        return np.tile(ctx['popularity'].astype(float), (len(ctx['users']), 1))
    if spec['kind'] == 'personal':
        return ctx['counts'].toarray()
    if spec['kind'] == 'recent':
        return recent_matrix(ctx, spec['half_life'], spec.get('play_weight', 'count')).toarray()
    if spec['kind'] == 'cf':
        return cf_scores(ctx, spec['metric'], spec['neighbors'], spec.get('representation', 'log1p'))
    if spec['kind'] == 'logistic':
        return ranker_scores(ctx, spec['C'])
    raise ValueError(f"Unknown model kind: {spec['kind']}")

def evaluate(scores, ctx, future, k=10):
    """Recall denominator includes out-of-catalog positives (honest end-to-end recall)."""
    if scores.shape != ctx['counts'].shape or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and match the historical user/catalog matrix.')
    if not 1 <= k <= len(ctx['artists']):
        raise ValueError('k must be between 1 and the catalog size.')
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
        repeat = target & hist
        rows.append(dict(user_index=int(u), recall=hits.sum()/len(target), precision=hits.mean(),
                         ndcg=float((hits/np.log2(np.arange(2,k+2))).sum()/ideal),
                         target_artists=len(target), covered_artists=len(target & catalog),
                         repeat_share=len(repeat)/len(target),
                         repeat_recall=len(set(rec)&repeat)/len(repeat) if repeat else np.nan,
                         novel_recall=len(set(rec)&novel)/len(novel) if novel else np.nan,
                         tail_recall=len(set(rec)&target&tail)/len(target&tail) if target&tail else np.nan,
                         history_events=int(ctx['history_sizes'][u]), recommendations=rec.tolist()))
    if not rows:
        raise ValueError('No eligible evaluation users; inspect history and windows.')
    per_user = pd.DataFrame(rows)
    summary = per_user[['recall','precision','ndcg','repeat_recall','novel_recall','tail_recall']].mean().to_dict()
    summary.update(users=len(rows), target_coverage=per_user.covered_artists.sum()/per_user.target_artists.sum(),
                   recommendation_coverage=len(set(a for r in per_user.recommendations for a in r))/len(catalog))
    return summary, per_user

def candidates():
    """Two trivial baselines, two course baselines, the temporal method and the learned ranker."""
    specs = [dict(name='Popularity',kind='popular'), dict(name='Personal counts',kind='personal')]
    specs += [dict(name=f'UserCF {m} k={k}',kind='cf',metric=m,neighbors=k)
              for m in ['cosine','jaccard'] for k in [20,40]]
    specs += [dict(name=f'Recency h={h}d',kind='recent',half_life=h) for h in [7,28,90]]
    specs += [dict(name=f'LogReg ranker C={C}',kind='logistic',C=C) for C in [0.01,0.1,1.0]]
    return specs

def compare(ctx,future,specs):
    results,details = [],{}
    for spec in specs:
        print('Evaluating:',spec['name'],flush=True)
        start = perf_counter()
        scores = model_scores(ctx,spec)
        scored = perf_counter()
        summary,users = evaluate(scores,ctx,future)
        summary.update(score_seconds=scored-start, evaluation_seconds=perf_counter()-scored,
                       score_matrix_mib=scores.nbytes/(1024**2))
        results.append(dict(model=spec['name'],**summary)); details[spec['name']] = users
    return pd.DataFrame(results).set_index('model'),details

def paired_bootstrap(a,b,metric='ndcg', repetitions=2000):
    if set(a.user_index) != set(b.user_index) or len(a) == 0:
        raise ValueError('Paired comparisons require the same nonempty set of users.')
    merged = a[['user_index',metric]].merge(b[['user_index',metric]],on='user_index',validate='one_to_one')
    d = (merged[f'{metric}_x']-merged[f'{metric}_y']).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = rng.choice(d,size=(repetitions,len(d)),replace=True).mean(axis=1)
    return dict(mean_difference=float(d.mean()), ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),users=len(d))

def summary_table(metrics, details, reference, metric='ndcg', repetitions=2000):
    """One row per model: its score with a user-bootstrap interval, other metrics, and the paired
    difference from `reference`. Per-model intervals overlap far more than paired ones because
    they include between-user difficulty, which pairing cancels."""
    rng = np.random.default_rng(SEED)
    rows = []
    for name in metrics.sort_values(metric).index:
        x = details[name][metric].to_numpy()
        boot = rng.choice(x, size=(repetitions, len(x)), replace=True).mean(axis=1)
        diff = paired_bootstrap(details[name], details[reference], metric, repetitions)
        rows.append({'model': name, metric: x.mean(), f'{metric}_ci_low': np.quantile(boot, .025), f'{metric}_ci_high': np.quantile(boot, .975),
                     **metrics.loc[name, ['precision','recall','repeat_recall','novel_recall','tail_recall']].to_dict(),
                     f'diff_vs_{reference}': diff['mean_difference'], 'diff_ci_low': diff['ci_low'], 'diff_ci_high': diff['ci_high']})
    return pd.DataFrame(rows).set_index('model')


def load_artist_names():
    """Labels for EDA/examples only; names do not enter model features."""
    names = {}
    for split in ['train', 'valid', 'test']:
        source = pq.ParquetFile(ROOT / f'data/raw/{split}.gz.parquet')
        for batch in source.iter_batches(columns=['artist_index','artist_name'], batch_size=500000):
            unique = batch.to_pandas().drop_duplicates('artist_index')
            for ident, label in zip(unique.artist_index, unique.artist_name):
                names.setdefault(int(ident), str(label))
    return names


def feature_representation_study(ctx, future, validation):
    """Validation-only controlled diagnostics; not new headline baselines."""
    rows = []
    for representation in ['binary', 'count', 'log1p']:
        summary, _ = evaluate(cf_scores(ctx, 'jaccard', 20, representation), ctx, future)
        rows.append(dict(experiment='Jaccard neighbor-vote representation',
                         representation=representation, **summary))
    # Matched personal-history contrast isolates age weighting, unlike CF vs Recency.
    summary, _ = evaluate(ctx['counts'].toarray(), ctx, future)
    rows.append(dict(experiment='Personal-history age weighting', representation='uniform play counts', **summary))
    for half_life in [7, 28, 90]:
        existing = validation.loc[f'Recency h={half_life}d'].to_dict()
        rows.append(dict(experiment='Personal-history age weighting',
                         representation=f'exponential half-life={half_life} days', **existing))
    return pd.DataFrame(rows)


def duration_study(ctx, future, half_lives=(7, 28, 90)):
    """Validation-only: play counts vs inter-play listening-time proxies inside Recency."""
    specs = [dict(name=f'Recency h={h}d {w}', kind='recent', half_life=h, play_weight=w)
             for h in half_lives for w in ['count', 'gap', 'relative', 'no_skip']]
    table, details = compare(ctx, future, specs)
    table.insert(0, 'play_weight', [s['play_weight'] for s in specs])
    table.insert(0, 'half_life', [s['half_life'] for s in specs])
    return table, details
