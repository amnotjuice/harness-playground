"""Does early self-test progress in an agent session predict the session's final score?

Reads agent + evaluator sessions (one per run, all from the same starting scheduler)
and, for each session, the best self-test score after k self-tests and the
session's official final score. Never calls a model.

Conventions, fixed before data collection:
  - Scores are the harness score: geometric mean of priority-weighted latency (s).
  - "Best after k" includes the starting scheduler, so failed or worse early
    self-tests leave it at the starting score.
  - A final scheduler that fails evaluation is plotted at the starting score
    (what the search would keep) and marked; a session with no self-tests, or
    whose generation failed, has no x and is excluded and counted.
  - A session with fewer than k self-tests is excluded at that k and counted.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

from .config import ROOT
from .predict import spearman

MAX_K = 10
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
        start = arm['trajectory'][0]['score']
        by_path = {r['path']: r for r in arm['trajectory']}
        for r in arm['trajectory'][1:]:
            tests = [json.loads(p.read_text()) for p in (Path(r['path']) / 'internal_evals').glob('*/eval.json')]
            tests.sort(key=lambda t: t.get('started_at_utc', ''))
            scores = [t['score'] if t.get('ok') else None for t in tests]
            parent = by_path.get(r.get('parent'))
            parent_score = parent['score'] if parent else start
            row = {'run': run.name, 'condition': run.parent.name, 'iteration': r['iteration'],
                   'status': r['status'], 'start_score': parent_score,
                   'self_tests': len(scores), 'valid_self_tests': sum(s is not None for s in scores),
                   'final_score': r['score'] if r['score'] is not None else parent_score,
                   'final_invalid': r['score'] is None,
                   'excluded': None, 'cost_usd': r.get('observed_cost_usd'),
                   'minutes': r['elapsed_seconds'] / 60}
            if r['status'] == 'generation_failed':
                row['excluded'] = 'generation failed'
            elif not scores:
                row['excluded'] = 'no self-tests'
            best = parent_score
            for k in range(1, MAX_K + 1):
                if k <= len(scores) and scores[k - 1] is not None:
                    best = min(best, scores[k - 1])
                row[f'best_after_{k}'] = best if k <= len(scores) and not row['excluded'] else None
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


def points(rows, k, outcome='final'):
    """(x, y, final_invalid) for sessions that reached k self-tests."""
    out = []
    for r in rows:
        x = r[f'best_after_{k}']
        if x is None:
            continue
        y = r['final_score'] if outcome == 'final' else r['final_score'] / x
        out.append((x, y, r['final_invalid']))
    return out


def summarize(rows):
    usable = [r for r in rows if not r['excluded']]
    summary = {'sessions': len(rows), 'usable': len(usable),
               'excluded': {reason: sum(r['excluded'] == reason for r in rows)
                            for reason in {r['excluded'] for r in rows if r['excluded']}},
               'final_invalid': sum(r['final_invalid'] for r in usable), 'by_k': {}}
    for k in range(1, MAX_K + 1):
        pts = points(rows, k)
        result = fit([p[0] for p in pts], [p[1] for p in pts])
        entry = {'n': len(pts), 'short_sessions': len(usable) - len(pts)}
        if result:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            entry.update({key: result[key] for key in ('slope', 'intercept', 'r2', 'residual_sd', 'spearman')},
                         r2_ci=bootstrap(xs, ys, 'r2'), residual_sd_ci=bootstrap(xs, ys, 'residual_sd'))
        summary['by_k'][k] = entry
    return summary


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
    start = rows[0]['start_score'] if rows else None
    excluded = ', '.join(f'{count} {reason}' for reason, count in summary['excluded'].items()) or 'none'
    footnote = (f"{summary['usable']} of {summary['sessions']} sessions usable (excluded: {excluded}). "
                f"Final scheduler failed evaluation in {summary['final_invalid']}: plotted at the starting score, "
                f"hollow. Sessions with fewer than k self-tests are left out of that panel.")

    def scatter_panels(name, outcome, ylabel, title):
        fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.6), squeeze=False)
        for ax, k in zip(axes[0], ks):
            pts = points(rows, k, outcome)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            result = fit(xs, ys)
            if result:
                lo, hi = min(xs), max(xs)
                line = [result['intercept'] + result['slope'] * x for x in (lo, hi)]
                sd = result['residual_sd']
                ax.fill_between([lo, hi], [v - sd for v in line], [v + sd for v in line], color=band,
                                zorder=1, label='±1 residual SD')
                ax.plot([lo, hi], line, color=ink, linewidth=1.5, zorder=2, label='Least-squares fit')
                spread = f'{sd:.1f} s' if outcome == 'final' else f'{sd:.3f}'
                rho = f" · ρ = {result['spearman']:.2f}" if result['spearman'] is not None else ''
                stats = f"n = {result['n']} · R² = {result['r2']:.2f} · residual SD = {spread}{rho}"
            else:
                stats = ''
                ax.text(.5, .5, f'Not enough sessions with ≥ {k} self-tests\n(n = {len(pts)})',
                        transform=ax.transAxes, ha='center', va='center', color=ink2)
            for x, y, invalid in pts:
                ax.plot(x, y, marker='o', linestyle='none', markersize=7, zorder=3,
                        markerfacecolor=surface if invalid else blue, markeredgecolor=blue,
                        markeredgewidth=1.5 if invalid else 1)
            if start and outcome == 'final':
                ax.axhline(start, color=ink2, linestyle=':', linewidth=1)
                ax.annotate('starting score', (1, start), xycoords=('axes fraction', 'data'), ha='right',
                            va='bottom', fontsize=8, color=ink2)
            ax.set(xlabel=f'Best score after {k} self-test{"s" if k > 1 else ""} (s)', ylabel=ylabel)
            # Statistics sit in the title so they never cover points.
            ax.set_title(f'k = {k}' + (f'\n{stats}' if result else ''), fontsize=10)
            if result:
                ax.legend(frameon=False, fontsize=8, loc='lower right')
        fig.suptitle(title, x=.01, ha='left', fontsize=12, fontweight='bold')
        import textwrap
        fig.text(.01, .01, textwrap.fill(footnote, int(fig.get_figwidth() * 15)), fontsize=8, color=ink2,
                 ha='left', va='bottom')
        fig.tight_layout(rect=(0, .07, 1, .94))
        fig.savefig(out / name, dpi=200)
        plt.close(fig)

    scatter_panels('A_final_vs_early.png', 'final', 'Final score (s, lower is better)',
                   'Does the score after k self-tests predict where the session ends?')
    scatter_panels('C_further_improvement.png', 'further', 'Final ÷ best after k (< 1: kept improving)',
                   'Does early progress predict further progress after k?')

    # B: how the fit sharpens with k; one measure per panel.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, key, label in ((axes[0], 'residual_sd', 'Residual SD around the fit (s)'), (axes[1], 'r2', 'R²')):
        ks_ok = [k for k, e in summary['by_k'].items() if e.get(key) is not None]
        for k in ks_ok:
            e = summary['by_k'][k]
            ci = e.get(f'{key}_ci')
            if ci:
                ax.plot([k, k], ci, color=band, linewidth=6, solid_capstyle='round', zorder=1)
            ax.plot(k, e[key], marker='o', markersize=7, color=blue, zorder=2)
            ax.annotate(f"n={e['n']}", (k, e[key]), xytext=(0, 9), textcoords='offset points', ha='center',
                        fontsize=7.5, color=ink2)
        if ks_ok:
            ax.plot(ks_ok, [summary['by_k'][k][key] for k in ks_ok], color=blue, linewidth=1.5, zorder=2)
        ax.set(xlabel='Checkpoint k (self-tests)', ylabel=label, xticks=range(1, MAX_K + 1))
        if key == 'r2':
            ax.set_ylim(0, 1.05)
    axes[0].set_title('Spread around the fit')
    axes[1].set_title('Variance explained')
    fig.suptitle('How early is early enough? Prediction quality by checkpoint (bands: 95% bootstrap)',
                 x=.01, ha='left', fontsize=12, fontweight='bold')
    fig.text(.01, .01, 'n falls with k because short sessions drop out; later checkpoints are biased toward '
             'long sessions.', fontsize=8, color=ink2, ha='left', va='bottom')
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(out / 'B_prediction_by_k.png', dpi=200)
    plt.close(fig)


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
    rows, summary = write_early_signal(args.directory, tuple(args.k))
    print(json.dumps({key: summary[key] for key in ('sessions', 'usable', 'excluded', 'final_invalid')}))
    print(f'{args.directory / "early_signal"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
