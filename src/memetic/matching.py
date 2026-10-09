"""Generator redesign, Phase 3: the actual matching pipeline -- Stage A
(structural routing, deterministic) and Stage B (semantic applicability,
LLM-judged). This is the module that turns (canonical, skill_bank) into the
final list of plausible (site, skill) pairs generation can use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from memetic.sites import Site, extract_meaningful_sites
from memetic.skills import Skill


@dataclass
class CandidatePair:
    site: Site
    skill: Skill


def match_sites_to_skills(canonical_source: str, skill_bank: List[Skill],
                          max_sites: Optional[int] = None) -> List[CandidatePair]:
    """STAGE A only -- cheap, deterministic, NO LLM call. Extracts every
    meaningful site from `canonical_source` (Phase 1) and, for each one,
    checks it against EVERY skill in `skill_bank`'s StructuralPrecondition.
    Returns every (site, skill) pair that survives -- these are candidates
    for Stage B's semantic judgment, NOT yet confirmed applicable.

    A single site can appear in multiple pairs (several skills may share a
    structural precondition loose enough to both match it) and a single
    skill can appear multiple times (matching several different sites in
    the same function) -- both are expected and intentional; Stage B is
    what actually narrows this down.

    CONFIRMED on real code: skills with a necessarily loose precondition
    (e.g. wrong_statistic, wrong_variable_reference -- no cheap structural
    feature reliably discriminates them, see skills.py's own comments) can
    dominate this list heavily (measured: 17/28 candidates on one realistic
    function). Do NOT tighten matches() to compensate -- see
    sample_candidates() below, which caps AFTER this stage instead, keeping
    matches() a pure boolean with no arbitrary per-skill hacks."""
    sites = extract_meaningful_sites(canonical_source, max_sites=max_sites)
    pairs = []
    for site in sites:
        for skill in skill_bank:
            if skill.structural.matches(site):
                pairs.append(CandidatePair(site=site, skill=skill))
    return pairs


def rank_candidates_by_embedding(pairs: List[CandidatePair], embedder,
                                 top_k: Optional[int] = None) -> List[CandidatePair]:
    """SECONDARY RANKER (design note Section 3: "Embeddings may be used only
    as a secondary ranker if many candidates survive"). Stage A's loose
    structural preconditions can return hundreds of (site, skill) pairs per
    canonical -- CONFIRMED on real data: a single loose-precondition skill
    matches every assignment/call/loop site, so on a 823-task held-out pool
    per-skill candidate counts ran into the thousands, and finding the few
    genuinely-applicable pairs meant judging (LLM) a large fraction of them.

    This orders the pairs by cheap embedding similarity between the SITE
    (its source text + enclosing statement) and the SKILL (its mechanism +
    semantic preconditions), so the most-likely-applicable pairs are judged
    FIRST and retrieval can cap to top_k Stage-B calls instead of judging
    everything. It does NOT decide applicability -- that stays entirely with
    Stage B (judge_skill_applicability); this only reorders the queue.

    Each distinct site and distinct skill is embedded ONCE (cached by id), so
    cost is O(#sites + #skills) encodes, not O(#pairs). Higher cosine first.
    On any embedding failure the input order is returned unchanged (ranking
    is an optimisation, never load-bearing)."""
    if not pairs:
        return []
    try:
        from memetic.embeddings import cosine_similarity
        site_vec = {}
        skill_vec = {}
        for p in pairs:
            sid = id(p.site)
            if sid not in site_vec:
                site_text = (p.site.source_text or "")
                if p.site.parent_text and p.site.parent_text != p.site.source_text:
                    site_text = site_text + " | " + p.site.parent_text
                site_vec[sid] = embedder.embed_one(site_text)
            kid = p.skill.skill_id
            if kid not in skill_vec:
                skill_vec[kid] = embedder.embed_one(
                    f"{p.skill.mechanism} {p.skill.semantic_preconditions}")
        scored = sorted(
            pairs,
            key=lambda p: cosine_similarity(site_vec[id(p.site)], skill_vec[p.skill.skill_id]),
            reverse=True)
        return scored[:top_k] if top_k else scored
    except Exception:
        return pairs[:top_k] if top_k else pairs


def sample_candidates(pairs: List[CandidatePair], max_per_skill: int = 3,
                      max_total: int = 12, seed: Optional[int] = None) -> List[CandidatePair]:
    """Caps the Stage-A output BEFORE it reaches Stage B's LLM judge --
    bounds cost without touching matches() itself. max_per_skill caps how
    many sites any single loose-precondition skill can contribute (directly
    addresses the wrong_statistic/wrong_variable_reference dominance found
    on real code); max_total is the final hard cap sent to Stage B.

    Deterministic with a given seed (for reproducible tests); random.sample
    without replacement within each skill's group, so different runs on
    the SAME canonical are not always identical -- matching the "diversify
    across the run" intent from the design note, rather than always picking
    the textually-first N sites."""
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    by_skill = defaultdict(list)
    for p in pairs:
        by_skill[p.skill.skill_id].append(p)

    capped = []
    for skill_id, group in by_skill.items():
        if len(group) > max_per_skill:
            capped.extend(rng.sample(group, max_per_skill))
        else:
            capped.extend(group)

    if len(capped) > max_total:
        # guarantee coverage FIRST (per the design note's own principle:
        # diversity via sampling/coverage, not left to chance) -- confirmed
        # needed on real data: an UNWEIGHTED random cut here dropped a rare
        # skill's only candidate entirely, purely by chance, even though
        # budget existed to keep it. Take 1 per distinct skill first (as
        # many as max_total allows), THEN fill remaining budget randomly
        # from what's left.
        by_skill_capped = defaultdict(list)
        for p in capped:
            by_skill_capped[p.skill.skill_id].append(p)
        guaranteed = []
        remaining_pool = []
        for skill_id, group in by_skill_capped.items():
            rng.shuffle(group)
            guaranteed.append(group[0])
            remaining_pool.extend(group[1:])
        if len(guaranteed) > max_total:
            capped = rng.sample(guaranteed, max_total)
        else:
            fill_n = max_total - len(guaranteed)
            capped = guaranteed + (rng.sample(remaining_pool, min(fill_n, len(remaining_pool)))
                                   if remaining_pool else [])
    return capped
