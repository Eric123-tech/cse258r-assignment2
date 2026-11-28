from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parents[1]
c=[]
def md(s):c.append(n.v4.new_markdown_cell(s))
def code(s):c.append(n.v4.new_code_cell(s))
md('''# CSE258R Assignment 2: Future artist listening prediction
**Question:** Given history before a common cutoff, which artists will an existing user listen to during the next **28 days**? Rank all training-derived candidate artists and return ten; previously heard artists remain eligible. This can support personalized listening queues. We compare two course-related collaborative-filtering baselines against a project method that ranks each user's time-decayed personal listening history.

Offline listening is a behavioral proxy, not satisfaction or causal evidence of recommendation benefit. An unobserved interaction is not a verified dislike. This workbook follows assignment2_fa26.pdf, PDF pages 8 and 10–13. The original test outcomes have already been inspected; additions below are descriptive diagnostics, not a fresh blind test. No additional test-driven model selection is performed. A presentation and independent interpretation remain student deliverables.''')
code('''import os, sys, json, inspect, platform
from pathlib import Path
ROOT=Path.cwd()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import pipeline as p
import analysis as analysis
np.random.seed(p.SEED)
REPORTS=ROOT/'reports'
REPORTS.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.figsize':(8,4),'axes.grid':True})
print('Python',platform.python_version(),'seed',p.SEED)
''')
md('''## 1. Prediction and evaluation design
**PDF question (p.10): Identify a predictive task on this dataset.**

**Answer.** For each historical user, rank artists by whether they will be heard at least once during the following 28 days. Inputs are historical user–artist exposures, play frequencies and event ages; the output is a top-ten artist list. A listening service could use this as a candidate queue for likely near-term interests. It includes familiar artists and does not claim to predict satisfaction, exact play counts or next-song order. Future observations provide an operational target that can be evaluated without collecting explicit ratings.

**PDF question: How will the predictive task be evaluated?**

**Answer.** Use macro NDCG@10 as the primary ranking measure because correct early suggestions are useful; supplement it with Precision@10, Recall@10 and repeat/novel breakdowns. Precision describes the fraction of the ten suggestions heard later, while recall describes the fraction of the future artist set retrieved. Neither is interchangeable with NDCG or classification accuracy. The chronology and shared candidate policy below define what information each model may use.

Two consecutive 28-day windows at the end of the observed collection period are validation and test. All earlier events train models. The collection endpoint determines the retrospective period; hyperparameters never use test outcomes. Timestamps follow the source clock without inferring local timezone.

Candidate artists require at least five historical distinct listeners. Eligible users require at least 100 history events and at least one future event; this conditions evaluation on future activity and does not model churn. Validation tunes settings; final models refit on training plus validation, rebuilding the catalog from that history.

Select models by macro **NDCG@10**, also report Recall@10, Precision@10, novel-artist recall, tail recall, recommendation coverage and target coverage. Relevance is binary: at least one future play. Recall denominators include out-of-catalog targets; ideal DCG includes them up to ten. All models use identical users/candidates within a split, with full-catalog evaluation and no sampled negatives. User-level averaging prevents prolific listeners dominating results.''')
md('''## 2. Data collection, preprocessing and exploratory analysis
The Last.fm scrobble logs were originally compiled by Òscar Celma. The Hugging Face derivative removes rows lacking MusicBrainz identifiers and adds integer IDs. Its default splits contain **disjoint users**. We combine their raw events and make a new chronological split for warm-user prediction. Optional demographics are unverified and are not model features.

**PDF question (p.10): Was preprocessing needed to obtain useful features?**

**Answer.** Yes. We turn event logs into historical artist sets, per-user artist counts, and play ages. We combine the default user-disjoint source splits because our task follows existing users through time; we do not treat those source split labels as model features. We preserve repeated plays because they measure listening frequency. Required-field nulls, identical records and timestamps outside the documented period are audited separately. Minimum historical support defines the candidate catalog; later target exclusions remain visible in coverage rather than disappearing from recall denominators.

We load user, artist, track and timestamp; remove null required fields and identical four-field rows, treating these as potential logging duplicates. The raw file includes isolated 2010/2013 timestamps and some records later than the documented May 5, 2009 collection endpoint. We conservatively exclude timestamps on or after 2009-05-05 (exclusive cutoff, no unverified endpoint-day assumption), before defining windows. Excluded events are saved for audit; this decision uses source provenance and timestamp validity, not predictive performance. Repeated plays at different times are preserved. Catalog filtering and all counts/similarities use history only. Data are for noncommercial use; acknowledge Last.fm and Celma. Download revision, sizes, URLs and SHA256 hashes are recorded in the manifest.''')
code('''manifest=json.loads((ROOT/'data/raw/manifest.json').read_text())
print('Dataset revision:',manifest['revision'])
display(pd.DataFrame(manifest['files'])[['file','bytes','sha256']])
events,audit=p.load_events()
display(audit)
audit.to_csv(REPORTS/'preprocessing_audit.csv',index=False)
artist_names=p.load_artist_names()
train,valid,test,boundaries=p.temporal_split(events)
print(json.dumps(boundaries,indent=2))
summary=pd.DataFrame([dict(split=name,events=len(f),users=f.user_index.nunique(),artists=f.artist_index.nunique(),start=f.timestamp.min(),end=f.timestamp.max()) for name,f in [('train',train),('validation',valid),('test',test)]])
display(summary)
summary.to_csv(REPORTS/'split_summary.csv',index=False)
(ROOT/'data/processed/split_config.json').write_text(json.dumps(boundaries,indent=2))
''')
md('''### Training distributions and decisions
Examine training-only distributions to motivate the model. Heavy-tailed activity motivates macro averaging; artist popularity motivates tail analysis; concentration of plays motivates studying personal listening history; changing interests motivate temporal weighting. The 28-day horizon and minimum-support thresholds are initial design choices, not selected on test performance. The last plotted training month is partial (March 1–9, 2009); its lower total is not evidence of a behavioral decline.''')
code('''user_events=train.groupby('user_index').size()
user_artists=train.groupby('user_index').artist_index.nunique()
listeners=train[['user_index','artist_index']].drop_duplicates().groupby('artist_index').size()
span=train.groupby('user_index').timestamp.agg(['min','max'])
span_days=(span['max']-span['min']).dt.total_seconds()/86400
pairs=train.groupby(['user_index','artist_index']).size()
display(pd.DataFrame(dict(events=user_events,artists=user_artists,history_days=span_days)).describe())
print('User–artist density:',len(pairs)/(len(user_events)*len(listeners)))
print('Fraction of plays beyond first user–artist occurrence:',1-len(pairs)/len(train))
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter, LogLocator
import matplotlib.dates as mdates
fig,axes=plt.subplots(2,3,figsize=(20,11),layout='constrained')
comma=FuncFormatter(lambda value,pos: f'{value:,.0f}')
for ax in axes.flat:
    ax.tick_params(axis='both',labelsize=12)
    ax.grid(alpha=.22)
    ax.title.set_fontsize(15)
    ax.xaxis.label.set_size(13); ax.yaxis.label.set_size(13)
axes[0,0].hist(user_events,bins=40,color='#2674a6',edgecolor='white')
axes[0,0].set(title='A. Listening activity per user',xlabel='Historical listening events per user (plays)',ylabel='Number of users in each bin (log scale)',yscale='log')
axes[0,0].xaxis.set_major_locator(MaxNLocator(5)); axes[0,0].xaxis.set_major_formatter(comma)
axes[0,1].hist(user_artists,bins=40,color='#2674a6',edgecolor='white')
axes[0,1].set(title='B. Artist diversity per user',xlabel='Distinct artists heard per user (artists)',ylabel='Number of users in each bin (log scale)',yscale='log')
axes[0,1].xaxis.set_major_formatter(comma)
axes[0,2].loglog(np.arange(1,len(listeners)+1),listeners.sort_values(ascending=False).values,color='#2674a6',linewidth=2)
axes[0,2].set(title='C. Artist popularity / long tail',xlabel='Artist rank by listener count (log scale)',ylabel='Distinct historical listeners (users; log scale)')
for ax in [axes[0,0],axes[0,1],axes[0,2]]:
    ax.yaxis.set_major_locator(LogLocator(base=10)); ax.yaxis.set_major_formatter(comma)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v,p:''))
axes[0,2].xaxis.set_major_locator(LogLocator(base=10)); axes[0,2].xaxis.set_major_formatter(comma)
axes[0,2].xaxis.set_minor_formatter(FuncFormatter(lambda v,p:''))
axes[1,0].hist(span_days,bins=30,color='#2674a6',edgecolor='white')
axes[1,0].set(title='D. Observed history length per user',xlabel='Time between first and last history event (days)',ylabel='Number of users in each bin')
axes[1,0].xaxis.set_major_formatter(comma)
monthly=train.groupby(train.timestamp.dt.to_period('M')).size()
axes[1,1].plot(monthly.index.to_timestamp(),monthly.values,color='#2674a6',linewidth=2)
axes[1,1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[1,1].tick_params(axis='x',labelrotation=45)
axes[1,1].yaxis.set_major_formatter(comma)
axes[1,1].set(title='E. Listening volume over time',xlabel='Calendar month (YYYY-MM)',ylabel='Listening events per month (plays)')
top10=pairs.groupby(level=0).nlargest(10).groupby(level=0).sum()/user_events
axes[1,2].hist(top10,bins=np.linspace(0,1,31),color='#2674a6',edgecolor='white')
axes[1,2].set(title='F. Concentration in personal top 10 artists',xlabel='Share of user plays from their top 10 artists (%)',ylabel='Number of users in each bin',xlim=(0,1))
axes[1,2].xaxis.set_major_formatter(PercentFormatter(xmax=1))
fig.suptitle(f'Training histories only: {len(user_events):,} users; {len(train):,} listening events',fontsize=18)
fig.savefig(REPORTS/'training_distributions.png',dpi=200,bbox_inches='tight'); plt.show()
''')
md('''### From exploratory findings to model features
There is no explicit rating column: a rating histogram is inapplicable. We instead analyze event counts and historical artist sets. We do not infer dislikes from missing events.

| Feature | Representation | EDA rationale | Where used |
|---|---|---|---|
| Artist exposure | 0/1 historical artist vector per user | User–artist matrix is sparse; many pairs lack observations | Jaccard/cosine user similarity |
| Repeated listening | Raw counts vs binary vs log(1+count), L1 normalized per neighbor | Repeat plays and concentrated interests; high counts could dominate | Controlled UserCF vote representations |
| Age of each play | Exponential weight in days relative to cutoff | Adjacent historical windows have incomplete artist-set overlap | Recency; matched validation contrast against uniform counts |
| Artist listener count | Distinct history users, minimum 5 | Long-tail support is uneven | Catalog eligibility and deterministic tie-breaking |

These are hypotheses justified by descriptive training evidence, not proof of predictive effects. Validation ablations below test the feature choices. Artist names are display labels only; optional demographics, text and audio are not predictive inputs.''')
code('''popular_artists=listeners.sort_values(ascending=False).head(10).rename('historical_listeners').reset_index()
popular_artists['artist']=popular_artists.artist_index.map(artist_names)
display(popular_artists[['artist','historical_listeners']])
popular_artists[['artist','historical_listeners']].to_csv(REPORTS/'training_popular_artists.csv',index=False)
''')
code('''ctx=p.make_context(train,boundaries['valid_start'])
x=ctx['counts']
display(pd.DataFrame([dict(history_users=x.shape[0],candidate_artists=x.shape[1],nonzero_pairs=x.nnz,density=x.nnz/(x.shape[0]*x.shape[1]),eligible_history_users=len(ctx['eligible']))]))
cut=pd.Timestamp(boundaries['valid_start'])
a=train[(train.timestamp>=cut-pd.Timedelta(days=56)) & (train.timestamp<cut-pd.Timedelta(days=28))].groupby('user_index').artist_index.agg(set)
b=train[train.timestamp>=cut-pd.Timedelta(days=28)].groupby('user_index').artist_index.agg(set)
common=a.index.intersection(b.index)
overlap=pd.Series([len(a[u]&b[u])/len(a[u]|b[u]) for u in common])
print('Artist-set overlap of adjacent historical 28-day windows:'); display(overlap.describe())
plt.hist(overlap,bins=30); plt.title('Artist-set overlap across adjacent historical 28-day windows'); plt.xlabel('Jaccard overlap of artist sets (0 = no overlap; 1 = identical)'); plt.ylabel('Number of users in each bin'); plt.show()
''')
code('''eda_answer,eda_statistics=analysis.exploratory_answers(
    train,user_events,user_artists,listeners,pairs,span_days,top10,overlap,artist_names)
display(Markdown(eda_answer))
eda_statistics.to_csv(REPORTS/'eda_findings.csv',index=False)
''')
md('''## 3. Baselines, the project method and a learned ranker
**PDF question (p.11): Which course models are relevant, and what are the baselines?**

**Answer.** Our main baseline is Jaccard UserCF and the second is cosine UserCF. Both directly use the observed user–artist interaction structure. They test whether similar listeners help predict future listening. We compare these with time-decayed personal preference to ask whether the user's recent history is more informative. Below them sit two trivial baselines that any useful model must beat, and above them a logistic-regression ranker that learns how to combine the signals.

**Trivial baseline 1: Popularity.** Recommend the ten artists with the most historical listeners to everyone. No personalization.

**Trivial baseline 2: Personal counts.** Recommend each user's ten most-played artists, ignoring when the plays happened. Neither trivial baseline has a parameter to tune.

**Baseline 1: Jaccard UserCF.** Let A(u) be the historical candidate artists heard by user u. User similarity is `|A(u) ∩ A(v)| / |A(u) ∪ A(v)|`. Select the k most similar other users, then sum their artist preferences weighted by similarity. Neighbor preferences are log(1 + play count), normalized per user. Tune k in {20,40} on validation.

**Baseline 2: Cosine UserCF.** Use cosine similarity of binary user–artist vectors and the same neighbor scoring procedure. This provides a second course-related baseline isolating the similarity choice. Tune k in {20,40}.

Direct slide evidence: [recommendation.pdf](https://drive.google.com/file/d/1VAt1C5Za5JCBXUkeRbqkH8jajgTdfmnl/view), PDF p.28, explicitly defines Jaccard for user or item sets; pp.29–31 discuss cosine; p.66 mentions using user similarities and changing weighting. The scoring adaptation for implicit artist ranking remains ours. The [2026 syllabus](https://cseweb.ucsd.edu/classes/fa26/cse258-001/) assigns PML chapter 4; the [official chapter 4 workbook](https://cseweb.ucsd.edu/~jmcauley/pml/code/chap4.html) explicitly defines Jaccard, cosine and similarity-based recommendation. Its worked retrieval example primarily compares items; our implementation adapts these principles to users and implicit artist plays. This is a course-method adaptation, not a claim that the exact implicit-feedback ranking pipeline appears verbatim in lecture.

**Project method: time-decayed personal preference (Recency).** Score each artist by summing `2 ** (-age_days / half_life)` over that user's historical plays. Tune half-life in {7,28,90} days on validation. This method emphasizes recent personal interests rather than neighbor preferences. The provided [temporal.pdf](https://drive.google.com/file/d/1YQdxP_boLqM24MEiXy65ii3Utiqp55R3/view), PDF pp.12–13, explicitly teaches exponential age weighting; we adapt it to calendar-day play ages. It is a course-inspired temporal heuristic applied to this task; it is not claimed as a new algorithm or an internal modification of Jaccard UserCF. Any improvement claim is empirical and specific to the evaluated dataset/window.

**Learned ranker: logistic regression over (user, artist) features.** Recency commits to one half-life and one way of counting. Instead we let a classifier (course chapter 3) learn the combination. Each candidate pair (u, a) — every artist u has heard plus the 20 top Jaccard-CF artists u has not heard — gets 21 features: play count, days since first/last play, decayed scores at half-lives 3/7/28/90/365 days, plays and distinct listening days in the last 28 days, distinct listening days overall, the artist's share of u's plays, u's overall and recent activity, the Jaccard CF score, artist popularity overall and in the last 28 days, and a heard-before flag. Count-like features are log(1+x) transformed and all are standardized. The label is whether u plays a in the next 28 days.

To train without touching the evaluation window, the ranker uses the *previous* 28 days inside the history: features from history before cutoff − 28 days, labels from [cutoff − 28 days, cutoff). For validation that is Feb 10 – Mar 9, 2009; for test it is the validation window. The regularization strength C ∈ {0.01, 0.1, 1} is chosen on validation NDCG@10. The trivial baselines and the ranker were added after the original comparison below had already been run on test; they were designed and tuned on validation only and evaluated on test once.

All methods except Popularity retain historical artists as candidates. Ties use historical listener popularity, then artist ID. CF excludes self-neighbors; all features precede the prediction cutoff. More neighbors smooth preferences; a shorter half-life gives more weight to recent behavior. Pairwise user similarities require O(U²) memory and full scores O(UI). Previously tested extra models and tables are preserved under `reports/archive_initial/` for audit, outside this focused comparison.''')
md(r'''### Inputs, outputs, optimization and alternatives
Let $c_{ua}$ count historical plays, $b_{ua}=1[c_{ua}>0]$, and $A(u)$ be user u's historical candidate-artist set. Let $N_k(u)$ contain the k highest-similarity other users. Neighbor features are $q_{va}=\log(1+c_{va})/\sum_j\log(1+c_{vj})$.

$$s_{CF}(u,a)=\sum_{v\in N_k(u)} sim(u,v)q_{va},\quad
sim_J(u,v)=\frac{|A(u)\cap A(v)|}{|A(u)\cup A(v)|},\quad
sim_C(u,v)=\frac{b_u^T b_v}{\|b_u\|_2\|b_v\|_2}.$$

$$s_{time}(u,a)=\sum_{e:\,u_e=u,\,a_e=a}2^{-(t_{cutoff}-t_e)/(h\text{ days})}.$$

For the learned ranker, with standardized feature vector $x_{ua}$ and label $y_{ua}=1$[u plays a in the next 28 days],

$$s_{LR}(u,a)=\theta^T x_{ua},\qquad \min_\theta\ -\sum_{(u,a)}\big[y_{ua}\log\sigma(\theta^T x_{ua})+(1-y_{ua})\log(1-\sigma(\theta^T x_{ua}))\big]+\frac{1}{2C}\|\theta\|_2^2 .$$

Artists outside u's candidate set rank below every candidate.

Popularity, personal counts, CF and Recency are scoring heuristics, with no learned regression coefficients and no gradient-descent loss; the ranker is the only model fitted by optimization. Hyperparameters k and h are selected by validation macro NDCG@10 grid search. h is a half-life: a play h days old contributes one-half the weight of a just-observed play. k controls the amount of neighborhood smoothing. Both differ from the fixed 28-day prediction horizon. Binary relevance discards how many future times an artist is played.

Neighborhood models fit observed user–artist behavior without audio/text features or explicit ratings. An explicit-rating MSE regressor would require a different target; raw plays are not star ratings. Logistic regression needs a labeled user–artist sample design; our ranker supplies one by restricting to candidate pairs and labeling them from an earlier window, which keeps the problem small (about 500k pairs) and avoids treating all ~20M unobserved pairs as negatives. An SVM would fit the same design but gives no probabilities and was not needed. Text models lack meaningful document features here. Implicit-feedback latent factors and sequence models are relevant alternatives discussed in related work, but are not implemented or claimed as evaluated. A large neural model is not necessary to answer the current question.

Noise includes possible erroneous timestamps, ambiguous duplicate logging, unverified demographics and selection from identifier filtering. Sparse long-tail artists may be excluded by minimum support. Grid search can still overfit validation; small grids and separate test refits reduce but do not eliminate this risk. The already-inspected test cannot be used to choose new representations.''')
code('''# Include source in the HTML so graders can inspect code without opening modules.
for fn in [p.make_context,p.recent_matrix,p.preference_matrix,p.cf_scores,p.pair_features,p.fit_ranker,p.ranker_scores,p.model_scores,p.evaluate]:
    display(Markdown('```python'+chr(10)+inspect.getsource(fn)+chr(10)+'```'))
''')
code('''specs=p.candidates()
validation,val_details=p.compare(ctx,valid,specs)
display(validation.sort_values('ndcg',ascending=False))
validation.to_csv(REPORTS/'validation_metrics.csv')
families={}
for spec in specs:
    family=spec['kind']+(' '+spec['metric'] if spec['kind']=='cf' else '')
    if family not in families or validation.loc[spec['name'],'ndcg']>validation.loc[families[family]['name'],'ndcg']:
        families[family]=spec
selected=list(families.values())
selected_best=max(selected,key=lambda s:validation.loc[s['name'],'ndcg'])['name']
(REPORTS/'selected_models.json').write_text(json.dumps(dict(metric='validation ndcg@10',best=selected_best,models=selected),indent=2))
print('Frozen validation-selected model:',selected_best)
''')
md('''### Which features does the ranker rely on?
Refit the selected ranker on the validation context (trained on Feb 10 – Mar 9 labels) and inspect its coefficients. Features are standardized, so magnitudes are comparable, but many are correlated (e.g. count and distinct listening days), so a coefficient is the effect *holding the others fixed*, not the feature's standalone usefulness.''')
code('''ranker_spec=next(s for s in selected if s['kind']=='logistic')
_,ranker,ranker_info=p.fit_ranker(ctx,ranker_spec['C'])
print('Training pairs:',ranker_info['pairs'],' positive rate:',round(ranker_info['positive_rate'],4))
coefficients=pd.Series(ranker.coef_[0],index=p.RANKER_FEATURES).sort_values()
coefficients.rename('coefficient').to_csv(REPORTS/'ranker_coefficients.csv')
fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
ax.barh(coefficients.index,coefficients,color=np.where(coefficients>0,'#2674a6','#c0504d'))
ax.axvline(0,color='black',lw=.8);ax.grid(False);ax.grid(axis='x',alpha=.25)
ax.set(title='Logistic-regression ranker coefficients (standardized features)',xlabel='Coefficient (positive = more likely to be played next 28 days)')
fig.savefig(REPORTS/'ranker_coefficients.png',dpi=180);plt.show()
coef=coefficients
display(Markdown(f"**Reading.** The strongest positive feature is the number of distinct days the user has listened to the artist "
    f"(`days_all`, {coef['days_all']:+.2f}), while raw play count is *negative* ({coef['count']:+.2f}) once days are held fixed: "
    "an artist played a little on many days is more likely to return than one binged on a single day. "
    f"Long-memory decay (`d365` {coef['d365']:+.2f}, `d90` {coef['d90']:+.2f}) and recency of the last play (`last` {coef['last']:+.2f}) matter more than "
    f"the 3–28 day decayed scores, which are largely redundant with them. The Jaccard CF score gets a small weight ({coef['cf']:+.2f}), "
    "consistent with neighbors adding little beyond the user's own history for this repeat-heavy task."))
''')
md('''### Validation-only feature and parameter analysis (slide p.11)
Keep the two headline baselines and their original selection grids. First vary neighbor vote representation while holding Jaccard similarity and k=20 fixed. Then compare uniform personal play counts with exponentially age-weighted counts, holding whose history is used fixed. The uniform-count row is a controlled ablation reference, not a third headline baseline. No new representation is selected for test from these results.

Parameter curves below reuse original validation scores. Timing columns measure this run's scoring and ranking/evaluation phases, not training epochs; dense score storage is reported in MiB. This is an illustrative resource measurement, not a hardware-controlled scalability benchmark.''')
code('''feature_study=p.feature_representation_study(ctx,valid,validation)
display(feature_study[['experiment','representation','ndcg','recall','precision']])
feature_study.to_csv(REPORTS/'validation_feature_representations.csv',index=False)
analysis.plot_feature_study(feature_study,REPORTS);plt.show()
analysis.plot_parameter_choices(validation,specs,REPORTS);plt.show()
display(Markdown(analysis.feature_interpretation(feature_study)))
display(validation[['score_seconds','evaluation_seconds','score_matrix_mib']])
''')
md('''### Does listening time help? A validation-only duration feature
The dataset has no play-duration column: each row only says a track was scrobbled at a time. Scrobble timestamps mark when a track started, so the gap to the same user's next play is roughly how long the track ran plus any pause. We test whether this proxy adds information beyond play counts inside Recency, holding everything else fixed:

- **count** (current): every play has weight 1.
- **gap**: weight = gap to the next play, capped at 10 minutes and scaled to [0, 1]. Long pauses between sessions hit the cap.
- **relative**: gap divided by that track's median plausible gap (30 s–20 min), capped at 1, i.e. "did this play run as long as the track usually does?" This removes the bias toward artists with long songs.
- **no_skip**: plays followed by another play within 30 s get weight 0.

Gaps are computed within history only. A user's last play before the cutoff has no observable next play, so it keeps weight 1; nothing after the cutoff is used. Last.fm only records a scrobble after half the track or four minutes, so true skips are mostly missing from the log. We therefore expected small effects. This is an added exploratory analysis: it is selected and reported on validation only, and does not change the frozen test models.''')
code('''duration,duration_details=p.duration_study(ctx,valid)
duration.to_csv(REPORTS/'validation_duration_features.csv')
display(duration.pivot(index='play_weight',columns='half_life',values='ndcg').loc[['count','gap','relative','no_skip']].round(4))
duration_ci=pd.DataFrame([dict(half_life=h,play_weight=w,**p.paired_bootstrap(duration_details[f'Recency h={h}d {w}'],duration_details[f'Recency h={h}d count']))
                          for h in [7,28,90] for w in ['gap','relative','no_skip']])
display(duration_ci.round(4))
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
weights=['count','gap','relative','no_skip']
for h in [7,28,90]:
    ax.plot(weights,[duration.loc[f'Recency h={h}d {w}','ndcg'] for w in weights],marker='o',label=f'h={h} days')
ax.set(ylabel='Validation macro NDCG@10 (axis does not start at 0)',xlabel='Per-play weight inside Recency',title='Play counts vs listening-time proxies (validation only)')
ax.legend();ax.grid(False);ax.grid(axis='y',alpha=.25)
fig.savefig(REPORTS/'validation_duration_features.png',dpi=180);plt.show()
best=duration_ci.loc[duration_ci.mean_difference.idxmax()]
display(Markdown(f"**Result.** No duration proxy reliably beats plain play counts. The largest gain is **{best.play_weight}** at h={best.half_life}: "
    f"{best.mean_difference:+.4f} NDCG@10, 95% interval [{best.ci_low:.4f}, {best.ci_high:.4f}], which includes zero. "
    f"The *relative* version, which removes song-length bias, is never better than counts. Only "
    f"{(p.play_weights(ctx['history'],'no_skip')==0).mean():.2%} of history plays look like skips, which matches Last.fm's scrobble rule. "
    "Our reading: for artist-level prediction, play counts already capture how much someone listens; the gap mostly adds song length and pause noise. "
    "We keep the simpler count representation."))
''')
md('''## 4. Held-out evaluation and validity
Refit each validation-selected family setting using training plus validation; evaluate the final 28 days. Do not filter past artists, because repeats belong to the task. Tail artists are the bottom 80% of catalog artists ordered by training listener popularity; novel targets are artists outside each user's full observed history (including any historical artist excluded from the catalog). Repeat recall evaluates previously heard targets separately. Out-of-catalog positives remain in recall denominators; coverage exposes them.

Report paired user-bootstrap confidence intervals (2,000 repetitions) against the Jaccard UserCF baseline and descriptive activity groups. User bootstrap measures variation among observed users, not uncertainty across different time periods. Subgroup comparisons are exploratory, not multiple independent confirmatory tests. A paired NDCG difference interval entirely above zero supports a positive average difference for this user sample under the fixed protocol; it does not establish causal benefit or robustness across time periods. Recall can be low alongside high precision because users may hear many more than ten artists in a 28-day window. NDCG rewards correct items near the top; it is not a percentage accuracy. There are no verified negative labels, so ordinary classification accuracy is not the task metric.''')
code('''history=pd.concat([train,valid],ignore_index=True)
test_ctx=p.make_context(history,boundaries['test_start'])
test_metrics,test_details=p.compare(test_ctx,test,selected)
display(test_metrics.sort_values('ndcg',ascending=False))
test_metrics.to_csv(REPORTS/'test_metrics.csv')
for name,f in test_details.items():
    safe=''.join(ch if ch.isalnum() else '_' for ch in name)
    f.drop(columns='recommendations').to_csv(REPORTS/f'users_{safe}.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(13,4))
for ax,metric in zip(axes,['ndcg','recall']):
    test_metrics[metric].plot.barh(ax=ax); ax.set_title(metric+'@10'); ax.set_xlabel('Macro user average')
fig.tight_layout(); fig.savefig(REPORTS/'test_comparison.png',dpi=160); plt.show()
''')
code('''jaccard_name=next(s['name'] for s in selected if s.get('metric')=='jaccard')
intervals=pd.DataFrame([dict(model=s['name'],**p.paired_bootstrap(test_details[s['name']],test_details[jaccard_name])) for s in selected if s['name']!=jaccard_name])
display(intervals); intervals.to_csv(REPORTS/'paired_bootstrap.csv',index=False)
groups=[]
for name,f in test_details.items():
    f=f.copy()
    f['activity']=pd.cut(f.history_events,[0,1000,10000,np.inf],labels=['100–999','1000–9999','10000+'],right=False)
    g=f.groupby('activity',observed=True).agg(users=('user_index','size'),ndcg=('ndcg','mean'),recall=('recall','mean'),repeat_share=('repeat_share','mean'))
    g['model']=name; groups.append(g.reset_index())
groups=pd.concat(groups,ignore_index=True)
display(groups); groups.to_csv(REPORTS/'activity_groups.csv',index=False)
''')
md('''### Summary: every model on the same test users
One table for all six models, sorted by NDCG@10. The NDCG interval is each model's own user-bootstrap 95% interval; the last three columns are the *paired* difference from Recency, our project method and the strongest heuristic.

Read the two kinds of interval differently. Per-model intervals are wide because users differ a lot in how predictable they are, and that spread is shared by every model, so two models' intervals can overlap even when one is reliably better. The paired interval compares models on the same users, cancelling that shared spread; a paired interval that excludes zero is the evidence for a real difference.''')
code('''recency_name=next(s['name'] for s in selected if s['kind']=='recent')
summary=p.summary_table(test_metrics,test_details,recency_name)
summary.to_csv(REPORTS/'test_summary_table.csv')
display(summary.round(4))
''')
md('''### Error analysis: repeat targets, discovery and user differences
Separate repeat-target recall from novel-target recall to avoid calling successful repetition 'discovery'. The following three Recency cases are selected deterministically from low, median and high NDCG users to illustrate outcomes; they are not random or representative examples. Artist names only label already-computed predictions. These descriptions are exploratory, not causal explanations.''')
code('''recency_name=next(s['name'] for s in selected if s['kind']=='recent')
display(test_metrics[['repeat_recall','novel_recall','tail_recall','target_coverage']])
analysis.plot_user_differences(test_details[recency_name],test_details[jaccard_name],REPORTS);plt.show()
case_summary,case_recommendations=analysis.error_examples(test_ctx,test,test_details[recency_name],artist_names)
display(case_summary)
with pd.option_context('display.max_colwidth',80):display(case_recommendations)
# Keep pseudonymous individual-user diagnostics local, consistent with .gitignore.
case_summary.to_csv(REPORTS/'users_case_summary.csv',index=False)
case_recommendations.to_csv(REPORTS/'users_case_recommendations.csv',index=False)
''')
code('''r=test_metrics.loc[selected_best]
ci=p.paired_bootstrap(test_details[selected_best],test_details[jaccard_name])
text=f"Validation selected **{selected_best}**. Held-out NDCG@10 = **{r['ndcg']:.4f}**, Recall@10 = **{r['recall']:.4f}**, evaluated users = **{int(r['users'])}**. Target coverage = **{r['target_coverage']:.2%}**."
text+=chr(10)*2+f"Paired NDCG difference from the Jaccard UserCF baseline = **{ci['mean_difference']:.4f}**, user-bootstrap 95% interval **[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]**. An interval crossing zero does not establish directional improvement. The CF and Recency rows reproduce the original frozen comparison exactly. The validation-only matched feature contrast is the relevant evidence about age weighting itself; the test contrast additionally changes personal versus neighbor signal. Negative results should be explained without modifying the held-out protocol."
display(Markdown(text))
''')
md('''### Does the learned ranker beat Recency, and what does it cost?
The headline intervals above are against Jaccard UserCF. The more demanding comparison is the ranker against the strongest heuristic, Recency, on the same users. We check it on validation (where C was chosen) and on test (evaluated once).''')
code('''ranker_name=ranker_spec['name']
lr_vs_recency=pd.DataFrame([dict(split='validation',**p.paired_bootstrap(val_details[ranker_name],val_details[recency_name])),
                            dict(split='test',**p.paired_bootstrap(test_details[ranker_name],test_details[recency_name]))])
display(lr_vs_recency.round(4)); lr_vs_recency.to_csv(REPORTS/'ranker_vs_recency_bootstrap.csv',index=False)
cols=['ndcg','precision','recall','repeat_recall','novel_recall','tail_recall','recommendation_coverage']
display(test_metrics.loc[[ranker_name,recency_name],cols].round(4))
t=lr_vs_recency.set_index('split').loc['test'];a=test_metrics.loc[ranker_name];b=test_metrics.loc[recency_name]
display(Markdown(f"**Result.** The ranker improves NDCG@10 over Recency by **{t.mean_difference:+.4f}** on test "
    f"(95% interval [{t.ci_low:.4f}, {t.ci_high:.4f}]), matching the validation gain, and also improves precision and repeat recall. "
    f"The cost is the long tail: tail recall drops from {b.tail_recall:.4f} to {a.tail_recall:.4f}, likely because it rewards long-term, many-day listening "
    f"and recent artist popularity (`pop28`), which favor established artists over niche ones. Novel recall stays near zero ({a.novel_recall:.4f}), so the gain comes from ordering familiar artists better, not from discovery. "
    "Which model to prefer depends on the goal: the ranker for accurate short-term queues, Recency if exposure to niche artists matters. "
    "Because the test window had been seen before these models were added, this is a single post-hoc test check of a validation-chosen model rather than a pristine blind test."))
''')
code('''display(Markdown(analysis.results_answers(test_metrics,intervals,validation,groups)))''')
md('''### Limitations and next experiments
Around 1,000 historical users do not constitute millions of independent samples. Identifier filtering, old collection period, exact-duplicate removal, artist support thresholds and conditioning on future-active users affect generalization. One terminal window does not establish temporal robustness. Offline accuracy cannot measure causal satisfaction or discovery benefit. Review novel and tail recall before claiming discovery improvements. Future work: rolling-origin evaluations, confidence-weighted implicit factorization and analysis of different activity levels/horizons, while reserving a new held-out period for any new decisions.

## 5. Related work: ideas, datasets, and comparison with our findings

**PDF questions (p.12): Where did the data come from, what similar datasets/methods exist, which ideas did we borrow, and how do earlier findings relate to ours?**

**Answer.** The data originate in Celma's collection of Last.fm API listening logs. The original related 360K release has more users but aggregate artist play counts, so the timestamped 1K release is suited to our event-age task. We borrow neighborhood similarities and exponential age weighting from course materials. Implicit-feedback factorization motivates treating missing events as uncertain; sequence attention motivates considering relevant recent actions. Our results support useful temporal information in this task, but do not replicate either paper's optimization or evaluation. The table distinguishes these connections from direct numerical comparison.

| Source | Dataset / task and main idea | Connection and limits of our experiment |
|---|---|---|
| Celma / MTG, Last.fm dataset archive; Celma (2010), *Music Recommendation and Discovery in the Long Tail* | [Creator archive](https://zenodo.org/records/6090214): Last.fm 1K contains timestamped user–artist–song events from `user.getRecentTracks`; Last.fm 360K contains aggregate user–artist play counts from `user.getTopArtists`. The book provides music-discovery/long-tail context. | We choose 1K because temporal events support future-window prediction; 360K is larger in users but its released aggregate counts cannot directly reproduce our event-age protocol. Our derivative removes missing MusicBrainz IDs. Low novel/tail recall means aggregate repeat-listening performance is not evidence of better long-tail discovery. |
| McAuley, recommendation and temporal lectures | [Recommendation slides](https://drive.google.com/file/d/1VAt1C5Za5JCBXUkeRbqkH8jajgTdfmnl/view), pp.28–31,66: similarities and weighting adaptations. [Temporal slides](https://drive.google.com/file/d/1YQdxP_boLqM24MEiXy65ii3Utiqp55R3/view), pp.12–13: exponential age weighting. | Both UserCF and age weighting have course foundations. Our contribution is their task-specific implementation, controlled feature analysis and empirical interpretation, not invention of decay. Differences between binary, count and log-count features are measured on validation. |
| Hu, Koren & Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets* | [Author-hosted paper](https://yifanhu.net/PUB/cf.pdf): distinguishes implicit preference from confidence and fits a confidence-weighted factor model; evaluates television watching. | Motivates treating missing plays as uncertain and counts differently from explicit ratings. Our log-count voting is a heuristic, not their confidence-weighted optimization, so our results neither reproduce nor refute their factorization findings. |
| Kang & McAuley (2018), *Self-Attentive Sequential Recommendation* (SASRec) | [Paper](https://arxiv.org/abs/1808.09781): self-attention selects relevant historical actions to predict the next item, with reported comparisons to MC/CNN/RNN sequence methods. | An advanced sequential direction beyond fixed decay: uses item order and learned history relevance. Our unordered future 28-day artist-set target, repeat allowance, catalog and small user cohort differ. Good fixed-decay results motivate testing temporal relevance but do not establish equivalence to learned attention. SASRec is an influential advanced method, not a claim about the best 2026 model. |

We compare our models numerically only under our shared protocol. Literature scores use different targets, datasets, item granularity, negatives, filtering and evaluation candidates; there is no defensible direct numerical ranking against them. No contemporary state-of-the-art benchmark is claimed. The central empirical finding is specific to this historical repeat-permitting prediction task, while feature ablations and error analysis qualify why it may hold.

## Submission preparation
The assignment requires an exported readable HTML notebook plus a roughly 20-minute video; peer graders should not need to run code. `workbook.html` is exported with saved outputs. Presentation outline: task/evaluation 3 min; EDA 4 min; models 5 min; results 5 min; related work/limitations 3 min. The video and later peer-grading report remain to be completed. No video link is fabricated.''')
nb=n.v4.new_notebook(cells=c,metadata={'kernelspec':{'display_name':'Python (project .venv)','language':'python','name':'python3'}})
n.write(nb,R/'notebooks/assignment2.ipynb')
print('Built',len(c),'cells')
