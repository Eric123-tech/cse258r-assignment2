"""Plots and evidence for the detailed Assignment 2 presentation requirements."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_parameter_choices(validation, specs, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    for metric in ['jaccard', 'cosine']:
        group = sorted([s for s in specs if s.get('metric') == metric], key=lambda s:s['neighbors'])
        axes[0].plot([s['neighbors'] for s in group], [validation.loc[s['name'],'ndcg'] for s in group], marker='o',label=metric)
    axes[0].set(title='UserCF neighbor choice (validation only)',xlabel='Number of other users (neighbors)',ylabel='Macro NDCG@10',xticks=[20,40])
    axes[0].legend()
    group = sorted([s for s in specs if s['kind']=='recent'],key=lambda s:s['half_life'])
    axes[1].plot([s['half_life'] for s in group],[validation.loc[s['name'],'ndcg'] for s in group],marker='o')
    axes[1].set(title='Recency parameter choice (validation only)',xlabel='Exponential half-life (days)',ylabel='Macro NDCG@10',xticks=[7,28,90])
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(Path(output)/'validation_parameter_choices.png',dpi=180)
    return fig


def plot_feature_study(study, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
    for ax,(experiment,group) in zip(axes,study.groupby('experiment',sort=False)):
        labels=group.representation.str.replace('exponential half-life=','h=',regex=False)
        ax.bar(np.arange(len(group)),group.ndcg,color='#2674a6')
        ax.set(xticks=np.arange(len(group)),xticklabels=labels,title=experiment,ylabel='Validation macro NDCG@10',xlabel='Feature representation (all else held fixed)')
        ax.tick_params(axis='x',labelrotation=25)
        ax.grid(axis='y',alpha=.25)
    fig.savefig(Path(output)/'validation_feature_representations.png',dpi=180)
    return fig


def feature_interpretation(study):
    cf=study[study.experiment=='Jaccard neighbor-vote representation'].set_index('representation')
    personal=study[study.experiment=='Personal-history age weighting'].set_index('representation')
    uniform=personal.loc['uniform play counts','ndcg']
    recent=personal.loc['exponential half-life=28 days','ndcg']
    return (f"**Controlled validation evidence.** With Jaccard similarity and 20 neighbors fixed, "
            f"binary votes yield NDCG@10 {cf.loc['binary','ndcg']:.4f}, raw-count votes "
            f"{cf.loc['count','ndcg']:.4f}, and log-count votes {cf.loc['log1p','ndcg']:.4f}. "
            "These change neighbor vote strength while keeping the neighborhood definition fixed. "
            f"For the same personal-history signal, uniform counts yield {uniform:.4f} and "
            f"28-day exponential weighting {recent:.4f} (difference {recent-uniform:+.4f}). "
            "This matched contrast is more informative about age weighting than CF versus Recency, "
            "which also changes whose history is used. These are descriptive validation ablations, "
            "not extra headline baselines or newly selected test models.")


def error_examples(ctx, future, per_user, artist_names):
    """Deterministic low/median/high outcome cases; intentionally not a random sample."""
    ordered=per_user.sort_values(['ndcg','user_index']).reset_index(drop=True)
    selected=ordered.iloc[[0,len(ordered)//2,len(ordered)-1]]
    truth=future.groupby('user_index').artist_index.agg(set)
    summary=[]; recommendations=[]
    for label,(_,row) in zip(['low','median','high'],selected.iterrows()):
        u=int(row.user_index);target=truth.loc[u];history=ctx['history_sets'][u]
        summary.append(dict(case=label,user_index=u,history_events=row.history_events,
                            target_artists=len(target),repeat_target_share=len(target&history)/len(target),
                            out_of_catalog_targets=len(target-set(ctx['artists'])),ndcg=row.ndcg,recall=row.recall))
        for rank,artist in enumerate(row.recommendations,1):
            recommendations.append(dict(case=label,user_index=u,rank=rank,
                                        artist=artist_names.get(artist,str(artist)),
                                        heard_in_history=artist in history,heard_in_future=artist in target))
    return pd.DataFrame(summary),pd.DataFrame(recommendations)


def plot_user_differences(recency, baseline, output):
    merged=recency[['user_index','ndcg']].merge(baseline[['user_index','ndcg']],on='user_index',validate='one_to_one')
    difference=merged.ndcg_x-merged.ndcg_y
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.hist(difference,bins=30,color='#2674a6',edgecolor='white')
    ax.axvline(0,color='black',linestyle='--',label='Equal performance')
    ax.set(title='Individual-user changes: Recency minus Jaccard UserCF',xlabel='Difference in user NDCG@10 (positive favors Recency)',ylabel='Number of users in each bin')
    ax.legend();ax.grid(axis='y',alpha=.25)
    fig.savefig(Path(output)/'per_user_ndcg_differences.png',dpi=180)
    return fig


def exploratory_answers(train, user_events, user_artists, listeners, pairs,
                        span_days, top10, overlap, artist_names):
    """Answer the EDA prompt using training-only numbers, not generic plot descriptions."""
    artist_plays=pairs.groupby(level='artist_index').sum()
    head_size=max(1,int(np.ceil(len(artist_plays)*.01)))
    head_share=artist_plays.nlargest(head_size).sum()/len(train)
    single_share=(listeners==1).mean()
    repeat_fraction=1-len(pairs)/len(train)
    density=len(pairs)/(len(user_events)*len(listeners))
    top_artist=int(listeners.idxmax())
    top_name=artist_names.get(top_artist,str(top_artist))
    facts={
        'training_events':len(train), 'training_users':len(user_events),
        'training_artists':len(listeners), 'nonzero_user_artist_pairs':len(pairs),
        'user_artist_density':density, 'median_events_per_user':float(user_events.median()),
        'mean_events_per_user':float(user_events.mean()), 'max_events_per_user':int(user_events.max()),
        'median_artists_per_user':float(user_artists.median()),
        'single_listener_artist_fraction':float(single_share),
        'top_one_percent_artists_play_fraction':float(head_share),
        'repeated_user_artist_event_fraction':float(repeat_fraction),
        'median_personal_top10_play_fraction':float(top10.median()),
        'median_history_span_days':float(span_days.median()),
        'historical_window_overlap_users':len(overlap),
        'mean_historical_artist_set_jaccard':float(overlap.mean()),
        'median_historical_artist_set_jaccard':float(overlap.median())}
    answer=f'''### Answer to the PDF question: exploratory analysis

> Perform an exploratory analysis of this dataset to identify interesting phenomena. (Assignment slides, pp.8–9)

**Answer.** The main training-data phenomena are uneven user activity, a long-tailed artist catalog, extensive repeat listening, and incomplete overlap between adjacent listening windows. These findings motivate both collaborative features and a personal temporal representation. They do not establish which model will predict best; that requires validation.

**What are the basic properties?** Before the validation cutoff, there are **{len(train):,} plays, {len(user_events):,} users and {len(listeners):,} distinct artists**, covering **{train.timestamp.min():%Y-%m-%d} to {train.timestamp.max():%Y-%m-%d}**. There are **{len(pairs):,} observed user–artist pairs**, occupying only **{density:.3%}** of possible pairs. These are scrobble events, not explicit ratings; a star-rating distribution would not describe this dataset. The median user has heard **{user_artists.median():,.0f} artists**.

**Phenomenon 1 — a few users contribute far more activity (panel A).** The median user contributes **{user_events.median():,.1f} plays**, compared with a mean of **{user_events.mean():,.1f}** and a maximum of **{user_events.max():,}**. This asymmetry means counting every event equally in evaluation would emphasize prolific listeners. We therefore average ranking metrics per user and normalize each neighbor's voting features. The distribution motivates this choice; it does not imply that less-active users have weaker musical preferences.

**Phenomenon 2 — popularity is concentrated and many artists have little support (panel C).** **{single_share:.1%} of artists have only one training listener.** The most-played **1% of artists ({head_size:,} artists)** account for **{head_share:.1%} of plays**; this uses play count, whereas panel C orders artists by distinct listeners. The artist with the most distinct training listeners is **{top_name} ({int(listeners.max()):,} listeners)**. This concentration makes rare-artist similarity estimates uncertain. We use historical support filtering and report tail/novel recall and target coverage so that success on familiar popular artists is not mistaken for catalog-wide discovery. The five-listener threshold is a design choice, not a value established as optimal by this plot.

**Phenomenon 3 — repeat listening is prominent (panel F).** **{repeat_fraction:.2%} of plays occur after the first observation of that user–artist pair.** For the median user, the personal top ten artists account for **{top10.median():.1%} of plays**. These statistics justify preserving repeat plays as features and allowing previously heard artists in the prediction list. This repeat-event statistic concerns repeated *events within history*; it is not the probability that a future artist will be a repeat and is not prediction accuracy.

**Phenomenon 4 — adjacent historical windows do not contain the same artist sets.** Among **{len(overlap):,} users active in both preceding 28-day windows**, artist-set Jaccard overlap has mean **{overlap.mean():.3f}** and median **{overlap.median():.3f}**, where 0 means disjoint sets and 1 means identical sets. This is consistent with temporal variation or intermittent sampling of interests; it does not by itself prove preference drift. It motivates testing age weighting against uniform counts using the same personal-history signal on validation.

**Phenomenon 5 — observation duration and time coverage need care (panels D–E).** Median first-to-last training history length is **{span_days.median():,.1f} days**. Users therefore have different amounts of historical opportunity. Monthly event volume changes also mix behavior with cohort coverage. The final training month only includes March 1–9, 2009; its lower total is a partial-window effect and cannot be interpreted as a genuine collapse in listening.

**How does this answer guide the rest of the project?** Sparse user–artist exposure supports binary Jaccard/cosine similarity; repeat concentration motivates count-based features; activity imbalance motivates log compression and per-neighbor normalization; changing observed sets motivate a temporal comparison. The validation-only feature study tests these hypotheses rather than treating plots as proof of prediction quality. Conclusions apply to this historical user sample after identifier filtering, not to all music listeners.
'''
    return answer,pd.DataFrame([{'statistic':k,'value':v} for k,v in facts.items()])


def results_answers(test_metrics, intervals, validation, groups):
    """Direct answers to slides 10, 11 and 13 using existing results only."""
    j='UserCF jaccard k=20';c='UserCF cosine k=20';r='Recency h=28d'
    ci=intervals.set_index('model')
    jac20=validation.loc[j,'ndcg'];jac40=validation.loc['UserCF jaccard k=40','ndcg']
    low=groups[(groups.model==r)&(groups.activity=='100–999')].iloc[0]
    high=groups[(groups.model==r)&(groups.activity=='10000+')].iloc[0]
    return f'''### Direct answers: results, significance and parameter interpretation

> Which models worked, which did not, and which features were predictive? (Slides pp.11,13)

**Answer.** Recency produces test NDCG@10 **{test_metrics.loc[r,'ndcg']:.4f}**, compared with **{test_metrics.loc[j,'ndcg']:.4f}** for the main Jaccard baseline and **{test_metrics.loc[c,'ndcg']:.4f}** for cosine. Thus temporal personal history works better than these neighborhood implementations for this particular repeat-permitting task. This does not establish that collaborative filtering is generally ineffective. The matched validation contrast, not this between-model test contrast, specifically supports the contribution of age weighting. Log-count neighbor votes also outperform binary/raw-count votes on validation.

> How do you assess validity and confirm that predictions are significant? (Slide p.10)

**Answer.** All methods use the same historical information cutoff, catalog, eligible users and relevance rule. We select parameters on validation and use full-catalog test ranking, not sampled negatives. For **{int(test_metrics.loc[r,'users'])} test users**, paired bootstrap gives Recency minus Jaccard NDCG difference **{ci.loc[r,'mean_difference']:+.4f}**, with 95% interval **[{ci.loc[r,'ci_low']:.4f}, {ci.loc[r,'ci_high']:.4f}]**. It excludes zero and supports a positive average difference under this fixed evaluation. The cosine-minus-Jaccard interval **[{ci.loc[c,'ci_low']:.4f}, {ci.loc[c,'ci_high']:.4f}]** crosses zero, so we do not claim a reliable superiority of cosine. User bootstrap does not measure variation across different time periods. Since these test results have already been seen, new explanatory analyses are descriptive rather than a new confirmatory test.

> What do the parameters mean, and what can be concluded from them? (Slide p.13)

**Answer.** k is the number of other users contributing votes, not a learned feature coefficient. Jaccard k=20 achieves validation NDCG **{jac20:.4f}**, versus **{jac40:.4f}** for k=40, so adding more neighbors did not help in this small grid. h controls memory: a play 28 days old has half the weight of a current play when h=28. Validation selects 28 days among 7/28/90; this is not evidence that 28 is globally optimal or that the best half-life must equal the prediction horizon.

> What does the error analysis reveal? (Slide p.13)

**Answer.** Recency's repeat-target recall is **{test_metrics.loc[r,'repeat_recall']:.4f}**, whereas novel-target recall is **{test_metrics.loc[r,'novel_recall']:.6f}**. Its strength is predicting familiar artists, not discovery. Recency NDCG is **{low.ndcg:.4f}** for users with 100–999 history plays (**{int(low.users)} users**) and **{high.ndcg:.4f}** for users with at least 10,000 (**{int(high.users)} users**). These groups also differ in repeat behavior and other characteristics, so this association does not show that adding plays would causally improve accuracy. Catalog target coverage is **{test_metrics.loc[r,'target_coverage']:.2%}**, exposing targets no model could recommend. Low/median/high examples illustrate specific outcomes without claiming to represent the population.
'''
