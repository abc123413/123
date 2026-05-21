import re
from typing import List

from app.config import Settings


class QueryRewriterService:
    """Lightweight query rewriting for better recall."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rewrite(self, query: str) -> List[str]:
        normalized = " ".join(query.split())
        if not normalized:
            return []

        rewrites = [normalized]
        compact = re.sub(r"[，。！？；：,.!?;:]+", " ", normalized)
        compact = " ".join(compact.split())
        if compact and compact not in rewrites:
            rewrites.append(compact)

        if "怎么" in normalized or "如何" in normalized:
            candidate = normalized.replace("怎么", "流程").replace("如何", "步骤")
            if candidate not in rewrites:
                rewrites.append(candidate)

        if "区别" in normalized or "对比" in normalized:
            candidate = normalized.replace("区别", "差异").replace("对比", "比较")
            if candidate not in rewrites:
                rewrites.append(candidate)

        keywords = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", normalized)
        if len(keywords) >= 2:
            candidate = " ".join(keywords[:6])
            if candidate not in rewrites:
                rewrites.append(candidate)

        return rewrites[: self.settings.query_rewrite_max_queries]
