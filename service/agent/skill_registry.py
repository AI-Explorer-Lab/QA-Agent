from __future__ import annotations

from typing import Dict

from service.agent.skills import ALL_SKILLS, FactLookupSkill, SkillDefinition


class SkillRegistry:
    def __init__(self) -> None:
        self._skills = list(ALL_SKILLS)
        self._query_type_to_skill: Dict[str, SkillDefinition] = {}
        for skill in self._skills:
            for query_type in skill.query_types:
                self._query_type_to_skill[query_type] = skill

    def select_skill(self, query_type: str) -> SkillDefinition:
        return self._query_type_to_skill.get(query_type, FactLookupSkill)

    def list_skills(self) -> list[str]:
        return [skill.skill_name for skill in self._skills]


DEFAULT_SKILL_REGISTRY = SkillRegistry()
