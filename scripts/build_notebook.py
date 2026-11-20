from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parents[1]
c=[]
def md(s):c.append(n.v4.new_markdown_cell(s))
def code(s):c.append(n.v4.new_code_cell(s))
md('''# CSE258R Assignment 2: Future artist listening prediction
**Question:** Given history before a common cutoff, which artists will an existing user listen to during the next **28 days**? Rank all training-derived candidate artists and return ten; previously heard artists remain eligible. This can support personalized listening queues. We compare two course-related collaborative-filtering baselines against a project method that ranks each user's time-decayed personal listening history.

Offline listening is a behavioral proxy, not satisfaction or causal evidence of recommendation benefit. An unobserved interaction is not a verified dislike. This executed workbook is an initial experiment; a presentation and independent interpretation remain student deliverables.''')
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
np.random.seed(p.SEED)
REPORTS=ROOT/'reports'
REPORTS.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.figsize':(8,4),'axes.grid':True})
print('Python',platform.python_version(),'seed',p.SEED)
''')
md('''## 1. Prediction and evaluation design
Two consecutive 28-day windows at the end of the observed collection period are validation and test. All earlier events train models. The collection endpoint determines the retrospective period; hyperparameters never use test outcomes. Timestamps follow the source clock without inferring local timezone.

Candidate artists require at least five historical distinct listeners. Eligible users require at least 100 history events and at least one future event; this conditions evaluation on future activity and does not model churn. Validation tunes settings; final models refit on training plus validation, rebuilding the catalog from that history.

Select models by macro **NDCG@10**, also report Recall@10, Precision@10, novel-artist recall, tail recall, recommendation coverage and target coverage. Relevance is binary: at least one future play. Recall denominators include out-of-catalog targets; ideal DCG includes them up to ten. All models use identical users/candidates within a split, with full-catalog evaluation and no sampled negatives. User-level averaging prevents prolific listeners dominating results.''')
md('''## 2. Data collection, preprocessing and exploratory analysis
The Last.fm scrobble logs were originally compiled by Òscar Celma. The Hugging Face derivative removes rows lacking MusicBrainz identifiers and adds integer IDs. Its default splits contain **disjoint users**. We combine their raw events and make a new chronological split for warm-user prediction. Optional demographics are unverified and are not model features.

We load user, artist, track and timestamp; remove null required fields and identical four-field rows, treating these as potential logging duplicates. The raw file includes isolated 2010/2013 timestamps and some records later than the documented May 5, 2009 collection endpoint. We conservatively exclude timestamps on or after 2009-05-05 (exclusive cutoff, no unverified endpoint-day assumption), before defining windows. Excluded events are saved for audit; this decision uses source provenance and timestamp validity, not predictive performance. Repeated plays at different times are preserved. Catalog filtering and all counts/similarities use history only. Data are for noncommercial use; acknowledge Last.fm and Celma. Download revision, sizes, URLs and SHA256 hashes are recorded in the manifest.''')
code('''manifest=json.loads((ROOT/'data/raw/manifest.json').read_text())
print('Dataset revision:',manifest['revision'])
display(pd.DataFrame(manifest['files'])[['file','bytes','sha256']])
events,audit=p.load_events()
display(audit)
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
md('''## 3. Two course-related baselines and the project method
**Baseline 1: Jaccard UserCF.** Let A(u) be the historical candidate artists heard by user u. User similarity is `|A(u) ∩ A(v)| / |A(u) ∪ A(v)|`. Select the k most similar other users, then sum their artist preferences weighted by similarity. Neighbor preferences are log(1 + play count), normalized per user. Tune k in {20,40} on validation.

**Baseline 2: Cosine UserCF.** Use cosine similarity of binary user–artist vectors and the same neighbor scoring procedure. This provides a second course-related baseline isolating the similarity choice. Tune k in {20,40}.

Course evidence: the [2026 syllabus](https://cseweb.ucsd.edu/classes/fa26/cse258-001/) assigns PML chapter 4; the [official chapter 4 workbook](https://cseweb.ucsd.edu/~jmcauley/pml/code/chap4.html) explicitly defines Jaccard, cosine and similarity-based recommendation. Its worked retrieval example primarily compares items; our implementation adapts these principles to users and implicit artist plays. This is a course-method adaptation, not a claim that the exact UserCF experiment appears verbatim in lecture.

**Project method: time-decayed personal preference (Recency).** Score each artist by summing `2 ** (-age_days / half_life)` over that user's historical plays. Tune half-life in {7,28,90} days on validation. This method emphasizes recent personal interests rather than neighbor preferences. It is a standard temporal heuristic applied to this task; it is not claimed as a new algorithm or an internal modification of Jaccard UserCF. Any improvement claim is empirical and specific to the evaluated dataset/window.

All three methods retain historical artists as candidates. Ties use historical listener popularity, then artist ID. CF excludes self-neighbors; all features precede the prediction cutoff. More neighbors smooth preferences; a shorter half-life gives more weight to recent behavior. Pairwise user similarities require O(U²) memory and full scores O(UI). Previously tested extra models and tables are preserved under `reports/archive_initial/` for audit, outside this focused comparison.''')
code('''# Include source in the HTML so graders can inspect code without opening modules.
for fn in [p.make_context,p.recent_matrix,p.cf_scores,p.model_scores,p.evaluate]:
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
md('''## 4. Held-out evaluation and validity
Refit each validation-selected family setting using training plus validation; evaluate the final 28 days. Do not filter past artists, because repeats belong to the task. Tail artists are the bottom 80% of catalog artists ordered by training listener popularity; novel targets are artists outside each user's catalog history. Out-of-catalog positives remain in recall denominators; coverage exposes them.

Report paired user-bootstrap confidence intervals (2,000 repetitions) against the Jaccard UserCF baseline and descriptive activity groups. User bootstrap measures variation among observed users, not uncertainty across different time periods. Subgroup comparisons are exploratory, not multiple independent confirmatory tests.''')
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
code('''r=test_metrics.loc[selected_best]
ci=p.paired_bootstrap(test_details[selected_best],test_details[jaccard_name])
text=f"Validation selected **{selected_best}**. Held-out NDCG@10 = **{r['ndcg']:.4f}**, Recall@10 = **{r['recall']:.4f}**, evaluated users = **{int(r['users'])}**. Target coverage = **{r['target_coverage']:.2%}**."
text+=f"\\n\\nPaired NDCG difference from the Jaccard UserCF baseline = **{ci['mean_difference']:.4f}**, user-bootstrap 95% interval **[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]**. An interval crossing zero does not establish directional improvement. Negative results should be explained without modifying the held-out protocol."
display(Markdown(text))
''')
md('''### Limitations and next experiments
Around 1,000 historical users do not constitute millions of independent samples. Identifier filtering, old collection period, exact-duplicate removal, artist support thresholds and conditioning on future-active users affect generalization. One terminal window does not establish temporal robustness. Offline accuracy cannot measure causal satisfaction or discovery benefit. Review novel and tail recall before claiming discovery improvements. Future work: rolling-origin evaluations, confidence-weighted implicit factorization and analysis of different activity levels/horizons, while reserving a new held-out period for any new decisions.

## 5. Related work
- **Celma (2010), Music Recommendation and Discovery in the Long Tail.** Dataset attribution and music-discovery context. Our task allows repeats, so high future-listening accuracy need not imply successful discovery. [Creator-hosted dataset archive](https://zenodo.org/records/6090214); [Last.fm](https://www.last.fm/); [derivative dataset card](https://huggingface.co/datasets/matthewfranglen/lastfm-1k).
- **McAuley, Personalized Machine Learning**, chapters 4–5,7. Neighborhood recommendation, factor representations and temporal modeling form the course-aligned foundation. [Official textbook and workbooks](https://cseweb.ucsd.edu/~jmcauley/pml/).
- **Hu, Koren and Volinsky (2008), Collaborative Filtering for Implicit Feedback Datasets.** Preference and confidence are distinguished using a confidence-weighted factorization objective. That factorization objective is not implemented in this focused neighborhood-versus-recency experiment. [Author-hosted paper](https://www.chrisvolinsky.com/publications/17546.pdf).

Our numerical comparisons are internal under a common protocol. Published scores with different splits, filtering, candidates or targets are not directly comparable. Before final submission, read the references and develop a fuller dataset-specific related-work comparison in your own words.

## Submission preparation
The assignment requires an exported readable HTML notebook plus a roughly 20-minute video; peer graders should not need to run code. `workbook.html` is exported with saved outputs. Presentation outline: task/evaluation 3 min; EDA 4 min; models 5 min; results 5 min; related work/limitations 3 min. The video and later peer-grading report remain to be completed. No video link is fabricated.''')
nb=n.v4.new_notebook(cells=c,metadata={'kernelspec':{'display_name':'Python (project .venv)','language':'python','name':'python3'}})
n.write(nb,R/'notebooks/assignment2.ipynb')
print('Built',len(c),'cells')
