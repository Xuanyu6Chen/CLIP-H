"""Tables shown at the end of each phase: one row per hypothesis, the same layout everywhere.

Every table has the columns ``#``, ``hypothesis``, ``from``, then the numbers of that phase,
then ``status`` and ``reason``. ``from`` says where a hypothesis came from:

    N339      neuron 339
    G2        weak-neuron group 2
    R1        added by the reviewers in Phase 2C
    N12+N88   merged from two of the above
    ~N753     neuron 753 as a best guess: Phase 2B reworded it a lot (see ``match_rewrites``)
    ?         written in Phase 2B, or could not be traced

The origin of a text is kept in a plain dict (normalized text -> id) that each table function
updates, so the id follows a hypothesis through rewording (2B) and merging (2C). Nothing in
this file calls an LLM or writes a file.
"""

import ast
import difflib
import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from fidelity_gate import norm_hypothesis


# --- Number formats: one way to print each kind of number ------------------------------------
def fmt_share(x, digits=1) -> str:
    """0.287 -> '28.7%'."""
    return '' if x is None or pd.isna(x) else f'{float(x) * 100:.{digits}f}%'


def fmt_num(x, digits=2, sign=False) -> str:
    return '' if x is None or pd.isna(x) else f'{float(x):{"+" if sign else ""}.{digits}f}'


def fmt_q(x) -> str:
    """q- and p-values: three decimals, '<0.001' below that."""
    if x is None or pd.isna(x):
        return ''
    return '<0.001' if float(x) < 0.001 else f'{float(x):.3f}'


def fmt_rates(present, absent) -> str:
    """Outcome rate when the hypothesis is true / when it is not: '72% / 47%'."""
    return f'{"none" if pd.isna(present) else f"{present * 100:.0f}%"} / ' \
           f'{"none" if pd.isna(absent) else f"{absent * 100:.0f}%"}'


def fmt_odds(row) -> str:
    """Odds ratio with its 95% interval: '2.84 (2.15-3.75)'."""
    if pd.isna(row.get('odds_ratio')):
        return ''
    if pd.isna(row.get('or_ci_low')) or pd.isna(row.get('or_ci_high')):
        return f"{row['odds_ratio']:.2f}"
    return f"{row['odds_ratio']:.2f} ({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f})"


# --- Origin ids ------------------------------------------------------------------------------
def neuron_id(neuron_idx) -> str:
    """Pool row id: N<neuron> for a strong neuron, G<k> for weak-neuron group -k."""
    n = int(neuron_idx)
    return f'N{n}' if n >= 0 else f'G{-n}'


def origin_of(origins: Dict[str, str], text) -> str:
    return origins.get(norm_hypothesis(text), '?')


def _join_ids(ids: Sequence[str]) -> str:
    parts = [p for i in ids for p in str(i).split('+')]
    known = list(dict.fromkeys(p for p in parts if p and p != '?'))
    return '+'.join(known + (['?'] if '?' in parts and known else [])) if known else '?'


def _finish(df: pd.DataFrame, sort_by=None, ascending=True) -> pd.DataFrame:
    """Kept rows first, then dropped ones; a running number in the first column."""
    if len(df) == 0:
        return df
    out = df.copy()
    out['_gone'] = out['status'].astype(str).str.startswith(('dropped', 'merged', 'not traced'))
    by = ['_gone'] + ([sort_by] if sort_by else [])
    out = out.sort_values(by, ascending=[True] + ([ascending] if sort_by else []), kind='stable')
    out = out.drop(columns=[c for c in out.columns if c.startswith('_')]).reset_index(drop=True)
    out.insert(0, '#', range(1, len(out) + 1))
    return out


def show(df: pd.DataFrame, funnel: Optional[dict] = None, phase: str = '', n_in: Optional[int] = None,
         n_out: Optional[int] = None, note: str = ''):
    """Print the one-line result of a phase, record it for the funnel, display the table.

    ``funnel`` is a dict phase -> {'in', 'out', 'note'} that Phase 3B prints as one table.
    """
    if phase:
        line = phase + ':'
        if n_in is not None and n_out is not None:
            line += f' {n_in} in -> {n_out} out.'
        print(f'{line} {note}'.rstrip())
        if funnel is not None:
            funnel[phase] = {'in': n_in, 'out': n_out, 'note': note}
    if df is None or len(df) == 0:
        return
    styler = df.style.hide(axis='index').set_properties(**{'text-align': 'left', 'white-space': 'normal',
                                                           'vertical-align': 'top'})
    styler = styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])
    if 'status' in df.columns:
        gone = df['status'].astype(str).str.startswith(('dropped', 'merged', 'not traced'))
        styler = styler.apply(lambda row: ['color: #999999' if gone[row.name] else '' for _ in row], axis=1)
    try:
        from IPython.display import display
    except ImportError:   # outside Jupyter: plain text
        print(df.to_string(index=False))
        return
    display(styler)


# --- Phase 2A, 2A+, fidelity gate ------------------------------------------------------------
def table_2a(pool_df: pd.DataFrame, activations: np.ndarray, origins: Dict[str, str]) -> pd.DataFrame:
    """Strong neurons: the kept candidate of each, with its fidelity."""
    strong = pool_df[pool_df['neuron_idx'].astype(int) >= 0]
    fid = next((c for c in pool_df.columns if c.endswith('_fidelity_score')), None)
    sel = next((c for c in pool_df.columns if c.startswith('target_')), None)
    rows = []
    for _, r in strong.iterrows():
        n = int(r['neuron_idx'])
        origins[norm_hypothesis(r['interpretation'])] = neuron_id(n)
        rows.append({'hypothesis': r['interpretation'], 'from': neuron_id(n),
                     'fires on (train)': int((activations[:, n] > 0).sum()),
                     'selection score': fmt_num(r[sel], 3, sign=True) if sel else '',
                     'fidelity F1': fmt_num(r[fid]) if fid else '', '_f': r[fid] if fid else 0.0,
                     'status': 'candidate', 'reason': ''})
    return _finish(pd.DataFrame(rows), '_f', ascending=False)


def table_2a_plus(pool_df: pd.DataFrame, groups: Dict[str, dict], origins: Dict[str, str]) -> pd.DataFrame:
    """Weak-neuron groups: one row per grouped hypothesis."""
    rows = []
    for _, r in pool_df[pool_df['neuron_idx'].astype(int) < 0].iterrows():
        gid = neuron_id(r['neuron_idx'])
        g = groups.get(str(int(r['neuron_idx'])), {})
        members = g.get('neuron_idxs') or g.get('neurons') or []
        origins[norm_hypothesis(r['interpretation'])] = gid
        rows.append({'hypothesis': r['interpretation'], 'from': gid,
                     'theme': g.get('theme', ''), 'neurons': len(members),
                     'member neurons': ', '.join(str(int(m)) for m in members),
                     'fires on (train)': g.get('n_positives', ''),
                     'status': 'candidate', 'reason': ''})
    return _finish(pd.DataFrame(rows))


def table_gate(pool_df: pd.DataFrame, kept_df: pd.DataFrame, min_fidelity: float,
               origins: Dict[str, str]) -> pd.DataFrame:
    """Every pool row with what the fidelity gate did to it. ``kept_df`` = ``curate(pool_df, ...)[0]``."""
    fid = next((c for c in pool_df.columns if c.endswith('_fidelity_score')), None)
    kept_ids = set(kept_df['neuron_idx'].astype(int))
    kept_by_text = {norm_hypothesis(r['interpretation']): neuron_id(r['neuron_idx']) for _, r in kept_df.iterrows()}
    rows = []
    for _, r in pool_df.iterrows():
        n, f = int(r['neuron_idx']), (r[fid] if fid else np.nan)
        if n in kept_ids:
            status, reason = 'kept', ('no fidelity score (grouped)' if pd.isna(f) else '')
        elif not pd.isna(f) and f < min_fidelity:
            status, reason = 'dropped', f'fidelity below {min_fidelity:.2f}'
        else:
            status, reason = 'dropped', f"same text as {kept_by_text.get(norm_hypothesis(r['interpretation']), '?')}"
        rows.append({'hypothesis': r['interpretation'], 'from': neuron_id(n), 'fidelity F1': fmt_num(f),
                     '_f': -1.0 if pd.isna(f) else f, 'status': status, 'reason': reason})
    return _finish(pd.DataFrame(rows), '_f', ascending=False)


# --- Phase 2B --------------------------------------------------------------------------------
SURE_RATIO = 0.8    # wording at least this similar: the link is shown plainly
GUESS_RATIO = 0.6   # between the two: the link is shown with a leading "~" (best guess); below: no link


def match_rewrites(before: Sequence[str], after: Sequence[str], skip: Sequence[int] = ()):
    """For each text in ``after``: (index of the ``before`` text it was rewritten from, similarity).

    The refiner returns a plain list of hypotheses, so the link has to be recovered from the
    wording: identical texts first (similarity 1.0), then the most similar pairs, one to one,
    down to ``GUESS_RATIO``. A split or a merge is therefore not followed: one side keeps the
    link, the others count as new. ``skip`` lists ``before`` rows the critic dropped; they match
    identical text only. (None, 0.0) = nothing similar enough.
    """
    nb, na = [norm_hypothesis(x) for x in before], [norm_hypothesis(x) for x in after]
    result = [(None, 0.0)] * len(after)
    used, first = set(), {}
    for i, t in enumerate(nb):
        first.setdefault(t, i)
    for j, t in enumerate(na):
        if t in first and first[t] not in used:
            result[j] = (first[t], 1.0)
            used.add(first[t])
    open_after = [j for j in range(len(after)) if result[j][0] is None]
    allowed = [i for i in range(len(before)) if i not in set(skip)]
    ratio = {(j, i): difflib.SequenceMatcher(None, na[j], nb[i]).ratio() for j in open_after for i in allowed}
    for (j, i), r in sorted(ratio.items(), key=lambda kv: -kv[1]):
        if r < GUESS_RATIO:
            break
        if result[j][0] is None and i not in used:
            result[j] = (i, r)
            used.add(i)
    return result


def table_2b(trace: List[dict], final_texts: Sequence[str], origins: Dict[str, str]) -> pd.DataFrame:
    """Every hypothesis that entered self-refine and what it became, then the ones written in 2B.

    ``trace`` is the list saved in self_refine_trace.json: each round holds the hypotheses it
    started with, their share of the discovery sample and the critic's action. The link from a
    hypothesis to its rewrite is recovered from the wording (``match_rewrites``); a link that is
    only a best guess gets a leading "~" in ``from`` for the rest of the pipeline.
    """
    if not trace:
        return pd.DataFrame()
    final_texts = list(final_texts)
    rounds = [[a['hypothesis'] for a in t['actions']] for t in trace] + [final_texts]
    acts = [{a['hypothesis']: a for a in t['actions']} for t in trace]
    start = rounds[0]
    alive = [{i: 1.0} for i in range(len(start))]     # per start row: its rows in the current round -> weakest link so far
    critic = [[] for _ in start]
    why = ['' for _ in start]
    for level in range(len(rounds) - 1):
        cur, nxt = rounds[level], rounds[level + 1]
        for s, idxs in enumerate(alive):
            for i in idxs:
                a = acts[level].get(cur[i], {})
                act = str(a.get('critic_action', 'accept'))
                if a.get('niche_broaden'):
                    act = 'too rare' if act == 'accept' else act + ', too rare'
                if act != 'accept':
                    critic[s].append(act)
                    why[s] = why[s] or str(a.get('critic_feedback', ''))
        dropped = [i for i, h in enumerate(cur) if str(acts[level].get(h, {}).get('critic_action', '')) == 'drop']
        parent = match_rewrites(cur, nxt, skip=dropped)
        alive = [{j: min(r, idxs[p]) for j, (p, r) in enumerate(parent) if p in idxs} for idxs in alive]
    rows, traced = [], set()
    for s, h in enumerate(start):
        src = origin_of(origins, h)
        ends = sorted(alive[s])
        for j in ends:
            link = src if alive[s][j] >= SURE_RATIO or src.startswith('~') else '~' + src
            origins[norm_hypothesis(final_texts[j])] = link
            traced.add(j)
        told_to_drop = any(a.startswith('drop') for a in critic[s])
        if not ends:
            status = 'dropped' if told_to_drop else 'not traced'
            reason = why[s] if told_to_drop else 'no hypothesis with similar wording after 2B: rewritten beyond recognition, or left out by the refiner'
        else:
            same = norm_hypothesis(final_texts[ends[0]]) == norm_hypothesis(h)
            status, reason = ('unchanged' if same else 'reworded'), why[s]
        rows.append({'hypothesis': h, 'from': src,
                     'true for (discovery sample)': fmt_share(acts[0].get(h, {}).get('prevalence')),
                     'critic': ' -> '.join(dict.fromkeys(critic[s])) or 'accept',
                     'after 2B': '' if status == 'unchanged' else ' | '.join(final_texts[j] for j in ends),
                     '_o': {'unchanged': 0, 'reworded': 1}.get(status, 3),
                     'status': status, 'reason': reason})
    for j, h in enumerate(final_texts):
        if j not in traced:
            origins.setdefault(norm_hypothesis(h), '?')
            rows.append({'hypothesis': '', 'from': origin_of(origins, h), 'true for (discovery sample)': '', 'critic': '',
                         'after 2B': h, '_o': 2, 'status': 'new in 2B',
                         'reason': 'written by the refiner (a split, a merge or a new idea); not similar enough to one hypothesis that entered'})
    return _finish(pd.DataFrame(rows), '_o')


# --- Phase 2C+ -------------------------------------------------------------------------------
def _as_list(value) -> list:
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    try:
        out = json.loads(str(value))
    except Exception:
        try:
            out = ast.literal_eval(str(value))
        except Exception:
            out = []
    return list(out) if isinstance(out, (list, tuple)) else []


def table_2c(merged_df: pd.DataFrame, candidate_texts: Sequence[str], origins: Dict[str, str], *,
             n_reviewers: int, min_clinical_validity: int, leakage_drop: int) -> pd.DataFrame:
    """Every hypothesis in the review reply with the reviewers' scores and whether it is kept."""
    majority = n_reviewers // 2 + 1
    rows, n_added = [], 0
    for _, r in merged_df.iterrows():
        sources = [int(i) for i in _as_list(r.get('source_ids'))]
        if sources:
            src = _join_ids([origin_of(origins, candidate_texts[i]) for i in sources if 0 <= i < len(candidate_texts)])
        else:
            n_added += 1
            src = f'R{n_added}'
        origins[norm_hypothesis(r['interpretation'])] = src
        why = []
        if int(r['valid_votes']) < majority:
            why.append(f"only {int(r['valid_votes'])} of {n_reviewers} valid votes")
        if int(r['leakage_risk']) >= leakage_drop:
            why.append(f"leakage risk {int(r['leakage_risk'])}")
        if int(r['clinical_validity']) < min_clinical_validity:
            why.append(f"clinical validity {int(r['clinical_validity'])}")
        keep = bool(r['keep'])
        note = 'added by the reviewers' if not sources else ('merged by the reviewers' if len(sources) > 1 else '')
        rows.append({'hypothesis': r['interpretation'], 'from': src,
                     'votes valid-invalid': f"{int(r['valid_votes'])}-{int(r['invalid_votes'])}",
                     'clinical validity (0-3)': int(r['clinical_validity']),
                     'leakage risk (0-2)': int(r['leakage_risk']),
                     'expected direction': r.get('expected_direction', ''),
                     '_v': -int(r['clinical_validity']),
                     'status': 'kept' if keep else 'dropped',
                     'reason': note if keep else ('; '.join(why) or 'not kept')})
    return _finish(pd.DataFrame(rows), '_v')


# --- Phase 2D --------------------------------------------------------------------------------
def table_2d(labels: Sequence[str], M: np.ndarray, dedup_result: dict, origins: Dict[str, str]) -> pd.DataFrame:
    """Every hypothesis with the other hypothesis it overlaps most with (phi on the discovery sample)."""
    M = np.asarray(M, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.nan_to_num(np.corrcoef(M.T)) if M.shape[1] > 1 else np.zeros((1, 1))
    np.fill_diagonal(phi, -np.inf)
    merged_into = {m: c['representative'] for c in dedup_result.get('clusters', [])
                   for m in c['members'] if m != c['representative']}
    rows = []
    for k, h in enumerate(labels):
        j = int(np.argmax(phi[k])) if len(labels) > 1 else k
        rows.append({'hypothesis': h, 'from': origin_of(origins, h),
                     'true for (discovery sample)': fmt_share(M[:, k].mean()),
                     'highest overlap (phi)': fmt_num(phi[k, j]) if len(labels) > 1 else '',
                     'with': labels[j] if len(labels) > 1 else '', '_p': phi[k, j] if len(labels) > 1 else 0.0,
                     'status': 'merged' if h in merged_into else 'kept',
                     'reason': f'merged into: {merged_into[h]}' if h in merged_into else ''})
    out = _finish(pd.DataFrame(rows), '_p', ascending=False)
    if len(out):   # name the other hypothesis by its row number instead of repeating its text
        number = dict(zip(out['hypothesis'], out['#']))
        out['with'] = out['with'].map(lambda h: f'#{number[h]}' if h in number else '')
    return out


# --- Phase 2E --------------------------------------------------------------------------------
def table_2e(gate_tests: pd.DataFrame, summary: dict, kept: Sequence[str], origins: Dict[str, str], *,
             fdr_alpha: float, stability_threshold: float, estimability_floor: int,
             outcome: str = 'positive') -> pd.DataFrame:
    """Every hypothesis tested by the confirmation gate, with the three numbers it is judged on.

    ``outcome`` names the positive class in the column '<outcome> if true / if not'.
    """
    stability = summary.get('stability', {})
    kept = set(kept)
    rows = []
    for _, r in gate_tests.iterrows():
        h, st = r['hypothesis'], stability.get(r['hypothesis'])
        why = []
        if r['n_present'] < estimability_floor:
            why.append(f"true for only {int(r['n_present'])} (needs {estimability_floor})")
        if pd.isna(r['q_value_fdr']) or r['q_value_fdr'] > fdr_alpha:
            why.append(f'q above {fdr_alpha:.2f}')
        if st is not None and st < stability_threshold:
            why.append(f'stability below {stability_threshold:.2f}')
        ok = h in kept
        rows.append({'hypothesis': h, 'from': origin_of(origins, h),
                     'true for (confirmation)': int(r['n_present']),
                     f'{outcome} if true / if not': fmt_rates(r['positive_rate_when_present'], r['positive_rate_when_absent']),
                     'odds ratio (95% CI)': fmt_odds(r), 'q': fmt_q(r['q_value_fdr']),
                     'stability': fmt_num(st), '_q': 1.0 if pd.isna(r['q_value_fdr']) else r['q_value_fdr'],
                     'status': 'certified' if ok else 'dropped',
                     'reason': ('kept to reach the minimum set size: ' + '; '.join(why) if ok and why
                                else '' if ok else '; '.join(why) or 'not in the kept set')})
    return _finish(pd.DataFrame(rows), '_q')


# --- Phase 3A --------------------------------------------------------------------------------
def table_3a(hypotheses: Sequence[str], X_test: np.ndarray, association_df: pd.DataFrame,
             coefficients: Sequence[float], origins: Dict[str, str], outcome: str = 'positive') -> pd.DataFrame:
    """The final hypotheses on the test set, with their weight in the prediction model."""
    assoc = association_df.set_index('hypothesis') if len(association_df) else pd.DataFrame()
    X_test = np.asarray(X_test, dtype=float)
    rows = []
    for k, h in enumerate(hypotheses):
        r = assoc.loc[h] if h in assoc.index else {}
        rows.append({'hypothesis': h, 'from': origin_of(origins, h),
                     'true for (test)': fmt_share(X_test[:, k].mean()),
                     f'{outcome} if true / if not': (fmt_rates(r['positive_rate_when_present'], r['positive_rate_when_absent'])
                                                     if len(r) else ''),
                     'odds ratio (95% CI)': fmt_odds(r) if len(r) else '',
                     'q (test)': fmt_q(r['q_value_fdr']) if len(r) else '',
                     'model coefficient': fmt_num(coefficients[k], sign=True), '_c': -abs(float(coefficients[k])),
                     'status': 'in the model', 'reason': ''})
    return _finish(pd.DataFrame(rows), '_c')


# --- Metrics and funnel ----------------------------------------------------------------------
def metrics_table(metrics: dict) -> pd.DataFrame:
    """Held-out metrics, one row each: value, 95% bootstrap interval, what to compare it with."""
    ci = metrics.get('ci95', {})
    k, prev = metrics.get('n_predicted_positive'), metrics.get('prevalence')
    compare = {'auroc': '0.500 = chance',
               'auprc': f'{prev:.3f} = share of positives in test' if prev is not None else '',
               'f1': f'the {k} highest-scored test patients are predicted positive' if k is not None else '',
               'precision': 'same cut; equal to F1, because as many are predicted positive as are positive',
               'recall': 'same cut; equal to F1'}
    rows = []
    for name in ('auroc', 'auprc', 'f1', 'precision', 'recall'):
        if name not in metrics:
            continue
        lo_hi = ci.get(name)
        rows.append({'metric': name.upper() if name.startswith('au') else name.capitalize(),
                     'value': f'{metrics[name]:.3f}',
                     '95% interval': f'{lo_hi[0]:.3f} to {lo_hi[1]:.3f}' if lo_hi else '',
                     'compare with': compare[name]})
    return pd.DataFrame(rows)


def arms_table(metrics_by_arm: Dict[str, dict]) -> pd.DataFrame:
    """The same metrics for several arms side by side: 'value (low to high)'."""
    rows = []
    for name in ('auroc', 'auprc', 'f1'):
        row = {'metric': name.upper() if name.startswith('au') else name.capitalize()}
        for arm, m in metrics_by_arm.items():
            lo_hi = m.get('ci95', {}).get(name)
            row[arm] = f'{m[name]:.3f}' + (f' ({lo_hi[0]:.3f} to {lo_hi[1]:.3f})' if lo_hi else '')
        rows.append(row)
    return pd.DataFrame(rows)


def funnel_table(funnel: Dict[str, dict]) -> pd.DataFrame:
    """One row per phase: hypotheses in, hypotheses out, what happened."""
    return pd.DataFrame([{'phase': phase, 'in': '' if v['in'] is None else v['in'],
                          'out': '' if v['out'] is None else v['out'], 'what happened': v['note']}
                         for phase, v in funnel.items()])
