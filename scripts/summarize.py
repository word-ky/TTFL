import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['method'], r['context'], r['lr'], r['steps'])].append(r)
    aggregates = []
    lookup = {}
    for (method, context, lr, steps), data in groups.items():
        values = [r['accuracy'] for r in data]
        r = dict(method=method, context=context, lr=lr, steps=steps, n=len(data),
                 accuracy=mean(values), accuracy_sd=stdev(values) if len(values)>1 else 0.,
                 gain_pp=mean(r['gain_pp'] for r in data),
                 support_loss_before=mean(r.get('support_loss_before', 0) for r in data),
                 support_loss_after=mean(r.get('support_loss_after', 0) for r in data),
                 query_loss_before=mean(r['query_loss_before'] for r in data),
                 query_loss_after=mean(r['loss'] for r in data),
                 worst_class_accuracy=mean(r['worst_class_accuracy'] for r in data),
                 gamma_norm=mean(r['gamma_norm'] for r in data),
                 beta_norm=mean(r['beta_norm'] for r in data))
        aggregates.append(r)
        lookup[(method, context, lr, steps)] = r
    gates = []
    for (method, context, lr, steps), data in groups.items():
        if method != 'affine' or context != 'correct':
            continue
        correct = lookup[(method, context, lr, steps)]
        deltas = {ctx: correct['accuracy'] - lookup[(method, ctx, lr, steps)]['accuracy']
                  for meth, ctx, l, s in lookup if meth == method and l == lr and s == steps
                  and (ctx.startswith('wrong_') or ctx == 'shuffled')}
        paired = {}
        by_seed = {r['support_seed']: r['accuracy'] for r in data}
        for ctx in deltas:
            vals = [by_seed[r['support_seed']] - r['accuracy']
                    for r in groups[(method, ctx, lr, steps)]]
            paired[ctx] = {'mean': mean(vals), 'sd': stdev(vals) if len(vals)>1 else 0.,
                           'min': min(vals), 'max': max(vals)}
        gates.append(dict(lr=lr, steps=steps, gain_pp=correct['gain_pp'],
                          context_deltas_pp=deltas, paired_deltas_pp=paired,
                          pass_gate=correct['gain_pp'] >= 5 and min(deltas.values()) >= 2))
    return {'aggregates': aggregates, 'gates': gates,
            't001_exploratory_gate_pass': any(g['pass_gate'] for g in gates),
            'qualification': 'One backbone seed, five support draws. Exploratory same-query grid; independent confirmation not performed.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    report = summarize(json.loads(Path(args.input).read_text()))
    (root/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    rows = report['aggregates']
    with (root/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# T001 complete grid', '', report['qualification'], '',
             '| Method | Context | LR | Steps | Acc mean ± SD (%) | Gain pp |',
             '|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['method']} | {r['context']} | {r['lr']} | {r['steps']} | {r['accuracy']:.3f} ± {r['accuracy_sd']:.3f} | {r['gain_pp']:+.3f} |" for r in rows]
    lines += ['', 'Exploratory gate: '+str(report['t001_exploratory_gate_pass'])]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['gates'], indent=2))


if __name__ == '__main__':
    main()
