from __future__ import annotations

from typing import Dict, List, Optional

from service.agent.skills import ALL_SKILLS, FactLookupSkill, SkillDefinition


class SkillRegistry:
    def __init__(self) -> None:
        self._skills = list(ALL_SKILLS)
        self._query_type_to_skill: Dict[str, SkillDefinition] = {}
        self._name_to_skill: Dict[str, SkillDefinition] = {}
        for skill in self._skills:
            self._name_to_skill[skill.skill_name] = skill
            for query_type in skill.query_types:
                self._query_type_to_skill[query_type] = skill

    def select_skill(self, query_type: str) -> SkillDefinition:
        return self._query_type_to_skill.get(query_type, FactLookupSkill)

    def get_skill(self, skill_name: str) -> Optional[SkillDefinition]:
        return self._name_to_skill.get(skill_name)

    def get_skill_package(self, skill_name: str) -> Optional[Dict[str, object]]:
        skill = self.get_skill(skill_name)
        if skill is None:
            return None
        return skill.package_metadata()

    def skill_catalog(self) -> List[Dict[str, object]]:
        return [skill.package_metadata() for skill in self._skills]

    def list_skills(self) -> List[str]:
        return [skill.skill_name for skill in self._skills]


DEFAULT_SKILL_REGISTRY = SkillRegistry()
