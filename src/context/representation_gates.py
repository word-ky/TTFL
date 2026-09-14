"""T021 inherits T018 capture/safety exactly; separate predeclared regret gates."""
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']


def inherited_gates(matched,actual,source=None):
    # Verbatim T018R matched/real expressions; delta_vs_BBSE is intentionally unchanged.
    matched_pass={t:all(r['capture_median']>=.8 for r in matched if r['context']==t) for t in C[1:]}
    real_pass={t:all(r['capture']>=.8 for r in actual if r['context']==t) for t in C[1:]}
    real_safe=all(r['delta_vs_BBSE_pp']>=-.5 for r in actual if r['context']!='clean')
    ma=sum(matched_pass.values())>=3;ra=sum(real_pass.values())>=3 and real_safe
    result=dict(REP_MATCH_A=ma,REP_REAL_A=ra,matched_pass=matched_pass,real_pass=real_pass,real_regression_safety=real_safe,REP_SRC_A='NOT_EXECUTED')
    if source is not None:
        source_pass={t:all(r['capture']>=.8 for r in source if r['context']==t) for t in C[1:]}
        clean_safe=all(r['clean_delta_vs_zero']>=-.5 for r in source if r['context']=='clean')
        result.update(REP_SRC_A=sum(source_pass.values())>=3 and clean_safe,source_pass=source_pass,source_clean_safety=clean_safe)
    return result


def regret_gate(rows):
    context_pass={t:all(r['baseline_mean']>0 and r['new_mean']<=.85*r['baseline_mean'] and r['new_p90']<=1.05*r['baseline_p90'] for r in rows if r['context']==t) for t in C[1:]}
    return dict(passed=sum(context_pass.values())>=3,context_pass=context_pass)


def diagnosis(gates):
    l,h=gates['L'],gates['H']
    if l['REP_MATCH_A'] and l['REP_REAL_A']:return 'T021-L'
    if not l['REP_REAL_A'] and h['REP_MATCH_A'] and h['REP_REAL_A']:return 'T021-H'
    if not l['REP_MATCH_A'] and not h['REP_MATCH_A']:return 'T021-N'
    if (l['REP_MATCH_A'] or h['REP_MATCH_A']) and not l['REP_REAL_A'] and not h['REP_REAL_A']:return 'T021-M'
    return 'T021-X'
