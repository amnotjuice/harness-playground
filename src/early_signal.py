"""Does early self-test progress in an agent session predict the session's final score?

Reads agent + evaluator sessions (one per run, all from the same starting scheduler)
and, for each session, two early signals and the session's official final score.
Never calls a model.

Conventions, fixed before data collection:
  - Scores are the harness score: geometric mean of priority-weighted latency (s).
  - Primary signal: "best tested after k" = the best score among the session's
    first k self-tests, NOT floored at the starting score, so early exploration
    that tests worse than the start still shows. Self-tests that fail to simulate
    have no score; a session whose first k self-tests all failed has no value at k.
  - Second signal: lines of code changed between the starting scheduler and the
    version the agent first tested (available before any simulation result).
  - A final scheduler that fails evaluation is plotted at the starting score
    (what the search would keep) and marked; a session with no self-tests, or
    whose generation failed, has no signal and is excluded and counted.
  - A session that OpenCode ended by auto-rejecting a permission request (the
    agent reached outside its workspace) was cut off by the harness, not stopped
    by the agent; it is excluded and counted.
  - A session with fewer than k self-tests, or no valid one yet, is excluded at
    that k and counted.
  - Near a session's end the best tested score is often the final itself, so a
    second fit by k keeps only sessions with at least TESTS_LEFT more self-tests.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import textwrap
from pathlib import Path

from .config import ROOT
from .predict import spearman
from .signals import code_features

MAX_K = 10
TESTS_LEFT = 3
MIN_SESSIONS_PER_K = 20
SESSION_ARM = 'opencode'


def session_rows(directory):
    """One row per agent session in a run or sweep directory."""
    directory = Path(directory)
    manifest = directory / 'sweep.json'
    runs = ([Path(c['output']) for c in json.loads(manifest.read_text())['cells']]
            if manifest.exists() else [directory])
    rows = []
    for run in runs:
        if not (run / 'results.json').exists():
            continue
        arm = json.loads((run / 'results.json').read_text())['arms'].get(SESSION_ARM)
        if not arm or not arm['trajectory'] or arm['trajectory'][0]['score'] is None:
            continue
        by_path = {r['path']: r for r in arm['trajectory']}
        for r in arm['trajectory'][1:]:
            # Each self-test directory holds the exact code tested and its evaluation.
            tests = [(p.parent, json.loads(p.read_text()))
                     for p in (Path(r['path']) / 'internal_evals').glob('*/eval.json')]
            tests.sort(key=lambda t: t[1].get('started_at_utc', ''))
            scores = [t['score'] if t.get('ok') else None for _, t in tests]
            parent = by_path.get(r.get('parent')) or arm['trajectory'][0]
            start_code = (Path(parent['path']) / 'scheduler.py').read_text()
            first_code = tests[0][0] / 'scheduler.py' if tests else None
            row = {'run': run.name, 'condition': run.parent.name, 'iteration': r['iteration'],
                   'status': r['status'], 'start_score': parent['score'],
                   'self_tests': len(scores), 'valid_self_tests': sum(s is not None for s in scores),
                   'lines_changed_before_first_test': (code_features(first_code.read_text(), start_code)
                                                       ['code_diff_lines']
                                                       if first_code and first_code.exists() else None),
                   'final_score': r['score'] if r['score'] is not None else parent['score'],
                   'final_invalid': r['score'] is None,
                   'excluded': None, 'cost_usd': r.get('observed_cost_usd'),
                   'minutes': r['elapsed_seconds'] / 60}
            stderr = Path(r['path']) / 'stderr.txt'
            if r['status'] == 'generation_failed':
                row['excluded'] = 'generation failed'
            elif stderr.exists() and 'auto-rejecting' in stderr.read_text():
                row['excluded'] = 'cut off by a rejected permission'
            elif not scores:
                row['excluded'] = 'no self-tests'
            best = None
            for k in range(1, MAX_K + 1):
                if k <= len(scores) and scores[k - 1] is not None:
                    best = scores[k - 1] if best is None else min(best, scores[k - 1])
                row[f'best_tested_after_{k}'] = best if k <= len(scores) and not row['excluded'] else None
            rows.append(row)
    return rows


def fit(xs, ys):
    """Least-squares line, R², residual SD (n - 2 degrees of freedom) and Spearman ρ."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(e * e for e in residuals)
    return {'n': n, 'slope': slope, 'intercept': intercept,
            'r2': 1 - sse / syy if syy else None, 'residual_sd': math.sqrt(sse / (n - 2)),
            'spearman': spearman(xs, ys), 'residuals': residuals}


def bootstrap(xs, ys, key, draws=2000, seed=0):
    """95% percentile interval of a fit statistic over sessions resampled with replacement."""
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        idx = [rng.randrange(len(xs)) for _ in xs]
        result = fit([xs[i] for i in idx], [ys[i] for i in idx])
        if result and result[key] is not None:
            values.append(result[key])
    if len(values) < draws / 2:
        return None
    values.sort()
    return values[int(.025 * len(values))], values[int(.975 * len(values)) - 1]


def points(rows, signal, min_tests=0):
    """(x, y, final_invalid) for usable sessions with a value for `signal` (a column name)
    and at least `min_tests` self-tests."""
    out = []
    for r in rows:
        x = r[signal]
        if r['excluded'] or x is None or r['self_tests'] < min_tests:
            continue
        out.append((x, r['final_score'], r['final_invalid']))
    return out


def fit_entry(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    entry = {'n': len(pts)}
    result = fit(xs, ys)
    if result:
        entry.update({key: result[key] for key in ('slope', 'intercept', 'r2', 'residual_sd', 'spearman')},
                     r2_ci=bootstrap(xs, ys, 'r2'), residual_sd_ci=bootstrap(xs, ys, 'residual_sd'))
    return entry


def summarize(rows):
    usable = [r for r in rows if not r['excluded']]
    summary = {'sessions': len(rows), 'usable': len(usable),
               'excluded': {reason: sum(r['excluded'] == reason for r in rows)
                            for reason in {r['excluded'] for r in rows if r['excluded']}},
               'final_invalid': sum(r['final_invalid'] for r in usable), 'by_k': {}, 'by_k_tests_left': {},
               'lines_changed_before_first_test': fit_entry(points(rows, 'lines_changed_before_first_test')),
               'self_tests': fit_entry(points(rows, 'self_tests'))}
    for k in range(1, MAX_K + 1):
        entry = fit_entry(points(rows, f'best_tested_after_{k}'))
        entry['fewer_than_k_tests'] = sum(r['self_tests'] < k for r in usable)
        entry['no_valid_test_yet'] = sum(r['self_tests'] >= k and r[f'best_tested_after_{k}'] is None for r in usable)
        summary['by_k'][k] = entry
        summary['by_k_tests_left'][k] = fit_entry(points(rows, f'best_tested_after_{k}',
                                                         min_tests=k + TESTS_LEFT))
    return summary


def verdict(result):
    """A plain-language reading of a fit, for plot titles."""
    share = max(result['r2'], 0) * 100
    if result['r2'] < .1:
        return f'No clear relationship (explains {share:.0f}%)'
    if result['r2'] < .3:
        return f'Weak relationship (explains {share:.0f}%)'
    return f'Explains {share:.0f}% of the variation'


def plot(rows, summary, out, ks=(1, 5)):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tools/state/cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    surface, ink, ink2, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    blue, band = '#2a78d6', '#cde2fb'
    plt.rcParams.update({'figure.facecolor': surface, 'axes.facecolor': surface, 'savefig.facecolor': surface,
                         'axes.edgecolor': ink2, 'axes.labelcolor': ink2, 'xtick.color': ink2,
                         'ytick.color': ink2, 'text.color': ink, 'font.size': 10, 'axes.titlesize': 11,
                         'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.color': grid})
    usable = [r for r in rows if not r['excluded']]
    start = rows[0]['start_score']
    # Only what applies to this run: exclusions and invalid finals are mentioned when there are any.
    notes = [f"{len(usable)} of {len(rows)} agent sessions from one starting scheduler (score {start:.1f} s)."]
    if summary['excluded']:
        notes.append('Not shown: ' + ', '.join(f'{count} {reason}' for reason, count in summary['excluded'].items())
                     + '.')
    if summary['final_invalid']:
        notes.append(f"{summary['final_invalid']} final schedulers failed evaluation: shown hollow at the "
                     'starting score.')
    if usable:
        finals = [r['final_score'] for r in usable]
        notes.append(f'Lower scores are better. All shown sessions ended between {min(finals):.1f} and '
                     f'{max(finals):.1f} s.')
    footnote = ' '.join(notes)

    def panel(ax, pts, xlabel, heading, missing=''):
        """One scatter with its best-fit line and typical miss, a verdict in the title and numbers below it."""
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        result = fit(xs, ys)
        if result:
            lo, hi = min(xs), max(xs)
            line = [result['intercept'] + result['slope'] * x for x in (lo, hi)]
            sd = result['residual_sd']
            ax.fill_between([lo, hi], [v - sd for v in line], [v + sd for v in line], color=band,
                            zorder=1, label=f'Typical miss (±{sd:.1f} s)')
            ax.plot([lo, hi], line, color=ink, linewidth=1.5, zorder=2, label='Best-fit line')
            ci = bootstrap(xs, ys, 'r2')
            details = f"{result['n']} sessions" + (f' · 95% CI {ci[0] * 100:.0f}–{ci[1] * 100:.0f}%' if ci else '')
            heading = f'{heading}\n{verdict(result)}'
            ax.legend(frameon=False, fontsize=8, loc='best')
        else:
            details = f'{len(pts)} sessions'
            reason = (f'All {len(pts)} sessions have the same value:\nnothing to fit'
                      if pts and len(set(xs)) == 1 else f'Not enough sessions (n = {len(pts)})')
            ax.text(.5, .5, reason, transform=ax.transAxes, ha='center', va='center', color=ink2)
        for x, y, invalid in pts:
            ax.plot(x, y, marker='o', linestyle='none', markersize=7, zorder=3,
                    markerfacecolor=surface if invalid else blue, markeredgecolor=blue,
                    markeredgewidth=1.5 if invalid else 1)
        if missing:
            details += f'\n{missing}'
        ax.set(xlabel=xlabel, ylabel='Final score (s)')
        # The verdict leads; the numbers sit in smaller grey text so they never cover points.
        ax.set_title(heading, fontsize=10, pad=8 + 12 * (details.count('\n') + 1))
        ax.text(0, 1.015, details, transform=ax.transAxes, fontsize=8.5, color=ink2, va='bottom')

    def finish(fig, name, title, note):
        fig.suptitle(title, x=.01, ha='left', fontsize=12, fontweight='bold')
        note = textwrap.fill(note, int(fig.get_figwidth() * 15))
        fig.text(.01, .01, note, fontsize=8, color=ink2, ha='left', va='bottom')
        # Reserve room for every footnote line so it never runs into the axis labels.
        bottom = (.12 + .15 * (note.count('\n') + 1)) / fig.get_figheight()
        fig.tight_layout(rect=(0, bottom, 1, 1 - .4 / fig.get_figheight()))
        fig.savefig(out / name, dpi=200)
        plt.close(fig)

    # A: the main question.
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.8), squeeze=False)
    for ax, k in zip(axes[0], ks):
        e = summary['by_k'][k]
        missing = ', '.join(text for count, text in (
            (e['fewer_than_k_tests'], f"{e['fewer_than_k_tests']} with fewer than {k} self-tests"),
            (e['no_valid_test_yet'], f"{e['no_valid_test_yet']} with no successful self-test yet")) if count)
        panel(ax, points(rows, f'best_tested_after_{k}'),
              'Score of the first self-test (s)' if k == 1 else f'Best score among the first {k} self-tests (s)',
              'After 1 self-test' if k == 1 else f'After {k} self-tests',
              f'Not shown: {missing}' if missing else '')
    finish(fig, 'A_final_vs_early.png', 'Does the early self-test score predict where the session ends?', footnote)

    # B: does prediction improve with more self-tests?
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    shown = {k: e for k, e in summary['by_k_tests_left'].items()
             if e['n'] >= MIN_SESSIONS_PER_K and e.get('r2') is not None}
    for k, e in shown.items():
        if e.get('r2_ci'):
            ax.plot([k, k], [max(v, 0) * 100 for v in e['r2_ci']], color=band, linewidth=8,
                    solid_capstyle='butt', zorder=1)
        ax.plot(k, max(e['r2'], 0) * 100, marker='o', markersize=7, color=blue, zorder=3)
    if shown:
        ax.plot(list(shown), [max(e['r2'], 0) * 100 for e in shown.values()], color=blue, linewidth=1.5, zorder=2)
        best = max(shown, key=lambda k: shown[k]['r2'])
        heading = (f"At most {max(shown[best]['r2'], 0) * 100:.0f}% of the variation explained "
                   f'(after {best} self-test{"s" if best > 1 else ""})')
    else:
        heading = 'Not enough sessions at any checkpoint'
    ax.set(xlabel='Self-tests seen so far', ylabel='Variation in final score explained (%)', ylim=(0, 100),
           xticks=list(shown), xticklabels=[f"{k}\n{e['n']} sessions" for k, e in shown.items()])
    ax.tick_params(axis='x', labelsize=8.5)
    ax.set_title(heading, fontsize=10)
    finish(fig, 'B_prediction_by_k.png', 'Does the early score predict better after more self-tests?',
           'Variation explained (R²) by a best-fit line from the best score so far to the final score; bars '
           'are 95% bootstrap intervals. Each point uses only sessions with at least '
           f'{TESTS_LEFT} more self-tests to go, so the score so far is never just the final scheduler. '
           f'Points with fewer than {MIN_SESSIONS_PER_K} such sessions are not shown. ' + footnote)

    # D: a signal available before any simulation result.
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    panel(ax, points(rows, 'lines_changed_before_first_test'),
          'Lines changed before the first self-test', 'Size of the first edit')
    finish(fig, 'D_code_changed_before_first_test.png', 'Does the size of the first edit predict the final score?',
           'Lines changed = lines added plus removed between the starting scheduler and the first version the '
           'agent tested. ' + footnote)

    # E: known only at the end, but it shows how much persistence matters.
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    panel(ax, points(rows, 'self_tests'), 'Self-tests the agent ran in the session', 'Number of self-tests')
    finish(fig, 'E_self_tests.png', 'Do sessions that self-test more end better?',
           'Known only once the session ends, so this is not an early signal. ' + footnote)


def write_early_signal(directory, ks=(1, 5)):
    directory = Path(directory)
    rows = session_rows(directory)
    out = directory / 'early_signal'
    out.mkdir(exist_ok=True)
    summary = summarize(rows)
    if rows:
        with (out / 'sessions.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot(rows, summary, out, ks)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path, help='a sweep directory of session runs, or one run')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 5], help='checkpoints for the scatter plots')
    args = parser.parse_args(argv)
    _, summary = write_early_signal(args.directory, tuple(args.k))
    print(json.dumps({key: summary[key] for key in ('sessions', 'usable', 'excluded', 'final_invalid')}))
    print(f'{args.directory / "early_signal"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
