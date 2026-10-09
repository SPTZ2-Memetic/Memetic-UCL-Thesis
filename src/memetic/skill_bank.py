"""Generator redesign: the SkillBank -- the live, mutable, persistent
collection of Skill objects. This is the actual artifact the rest of the
system talks to:

  - Phase 3 matching (matching.py's match_sites_to_skills) reads from it
    (via .all()) to find (site, skill) candidates for a canonical function.
  - The offline distillation pass (scripts/probe_distillation.py) writes
    into it to bootstrap the initial bank from BugSourceBench.
  - The live self-play training loop will ALSO write into it, continuously
    -- per generator-redesign-note.md section 7, the exact same
    localize-then-distill machinery runs on every productive bug from both
    the skilled and skill-free generation streams, every round, updating
    existing skills' statistics (distinct_tasks, valid_applications,
    recent_use_count, solve_rate_ema, last_evaluated_round) or adding new
    ones. That is a fundamentally different shape from "run a script once,
    get a static list" -- a bank is keyed, queried, updated in place, and
    persisted incrementally across rounds, not just serialized once at the
    end of a one-shot probe. A bare List[Skill] plus two free functions
    (the previous shape of this code) had nowhere to put any of that, which
    is why this is its own class rather than staying bolted onto skills.py.

Dedup/near-duplicate merging across distillation runs (section 7's same/
specialization/new outcomes) is implemented in dedup.py, which calls
add_distilled() (genuinely new) or merge_manifestation() (same as
existing) -- this module doesn't decide that itself, only stores the
result. The live per-round stats-update methods (solve_rate_ema etc., per
generator-redesign-note.md section 6) are still NOT implemented, deferred
pending live fixer infra.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, Iterator, List, Optional

from memetic.distill import DistilledSkill, to_skill
from memetic.skills import Skill, SkillExample, StructuralPrecondition


class SkillBank:
    """A keyed, mutable collection of Skill objects, keyed by skill_id."""

    def __init__(self, skills: Optional[List[Skill]] = None):
        self._by_id: Dict[str, Skill] = {}
        for s in (skills or []):
            self._by_id[s.skill_id] = s

    # ---- read ---------------------------------------------------------
    def all(self) -> List[Skill]:
        """Every skill in the bank, as a plain list -- what
        match_sites_to_skills (matching.py) and anywhere else expecting a
        List[Skill] (e.g. SEED_SKILLS-shaped call sites) consume."""
        return list(self._by_id.values())

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._by_id.get(skill_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._by_id.values())

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._by_id

    # ---- write ----------------------------------------------------------
    def add(self, skill: Skill, *, overwrite: bool = False) -> Skill:
        """Insert a Skill under its own skill_id. Raises on an id collision
        unless overwrite=True -- a caller minting fresh ids per distilled
        result should never collide; a collision is a real id-generation
        bug, not something to paper over silently."""
        if not overwrite and skill.skill_id in self._by_id:
            raise ValueError(
                f"SkillBank already has a skill with id {skill.skill_id!r} "
                f"-- pass overwrite=True to replace it")
        self._by_id[skill.skill_id] = skill
        return skill

    def add_distilled(self, distilled: DistilledSkill, site_type: str,
                      skill_id: str) -> Skill:
        """to_skill() + add() in one call -- the path both
        scripts/probe_distillation.py and, later, the live self-play
        training loop use to turn one successfully-distilled case into a
        stored, matchable Skill.

        NOTE: this always creates a NEW skill unconditionally -- it does
        NOT check whether `distilled` is actually the same mechanism as
        something already in the bank. That decision (section 7's
        same/specialization/new outcomes) lives in dedup.resolve_skill(),
        which calls this only for the genuinely-new case and calls
        merge_manifestation() instead for the same-mechanism case. Callers
        distilling real data should go through dedup.resolve_skill(), not
        call this directly -- see that module for why (confirmed on real
        data: calling this directly for every case produced 200 skills
        from 200 cases, a 0% merge rate)."""
        skill = to_skill(distilled, site_type, skill_id)
        return self.add(skill)

    def merge_manifestation(self, skill_id: str, example: SkillExample) -> Skill:
        """Section 7's "same as existing skill" outcome: no new skill --
        log this case as another manifestation (both the plain diff string
        AND the full-provenance SkillExample -- see SkillExample's
        docstring for why both lists exist) on the existing one, and bump
        its statistics. Raises KeyError if skill_id isn't in the bank (a
        caller error, not something to paper over).

        distinct_tasks/valid_applications both +1 per merge -- this was
        already correct (a merge IS one more distinct, valid application);
        the bug was in creation (to_skill()) never counting the FOUNDING
        case, now fixed there -- see that function's docstring."""
        skill = self._by_id[skill_id]
        if example.diff and example.diff not in skill.manifestations:
            skill.manifestations.append(example.diff)
        skill.examples.append(example)
        skill.distinct_tasks += 1
        skill.valid_applications += 1
        return skill

    # ---- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        """Writes the bank to `path` as a JSON array of Skill dicts (via
        dataclasses.asdict, which recurses into the nested
        StructuralPrecondition automatically). Plain JSON, not pickle --
        human-inspectable/diffable, matching this project's "verify
        everything by hand first" pattern. Overwrites path if present."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in self.all()], f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SkillBank":
        """Inverse of save() -- reconstructs real Skill/StructuralPrecondition/
        SkillExample dataclass instances (not bare dicts -- asdict() flattens
        SkillExample into plain dicts on save just like StructuralPrecondition,
        so both need reconstructing here the same way), so a loaded bank
        behaves identically to one built from SEED_SKILLS or distilled
        fresh."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        skills = []
        for d in raw:
            d = dict(d)   # shallow copy, don't mutate the parsed JSON in place
            d["structural"] = StructuralPrecondition(**d["structural"])
            d["examples"] = [SkillExample(**e) for e in d.get("examples", [])]
            skills.append(Skill(**d))
        return cls(skills)
