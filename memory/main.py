from __future__ import annotations

import re
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter


# -----------------------------
# Helpers: super-light semantic search (bag-of-words cosine)
# -----------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def bow_vector(text: str) -> Counter:
    return Counter(tokenize(text))

def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0) for t in a)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# -----------------------------
# Memory data model
# -----------------------------
@dataclass
class MemoryItem:
    id: str
    user_id: str
    category: str            # "semantic" | "episodic" | "procedure"
    text: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # cached vector for faster search
    _vec: Counter = field(default_factory=Counter, repr=False)


# -----------------------------
# Memory store with recency scoring + simple conflict resolution
# -----------------------------
class MemoryStore:
    def __init__(self):
        self._items: List[MemoryItem] = []

    def add(
        self,
        user_id: str,
        text: str,
        category: str = "semantic",
        now: Optional[datetime] = None,
        **metadata: Any,
    ) -> MemoryItem:
        now = now or datetime.utcnow()
        item = MemoryItem(
            id=str(uuid.uuid4()),
            user_id=user_id,
            category=category,
            text=text.strip(),
            created_at=now,
            metadata=dict(metadata),
        )
        item._vec = bow_vector(item.text)
        self._items.append(item)
        return item

    def update(self, item_id: str, new_text: str, now: Optional[datetime] = None, **metadata_updates: Any) -> MemoryItem:
        now = now or datetime.utcnow()
        item = self._get(item_id)
        item.text = new_text.strip()
        item.updated_at = now
        item.metadata.update(metadata_updates)
        item._vec = bow_vector(item.text)
        return item

    def delete(self, item_id: str) -> None:
        self._items = [x for x in self._items if x.id != item_id]

    def search(
        self,
        user_id: str,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
        now: Optional[datetime] = None,
        half_life_days: float = 30.0,   # recency decay
    ) -> List[Tuple[MemoryItem, float, float, float]]:
        """
        Returns list of (item, final_score, sim_score, recency_score)
        final_score = 0.75 * similarity + 0.25 * recency
        """
        now = now or datetime.utcnow()
        qvec = bow_vector(query)

        candidates = [
            x for x in self._items
            if x.user_id == user_id and (category is None or x.category == category)
        ]

        scored = []
        for item in candidates:
            sim = cosine(qvec, item._vec)

            # recency score using exponential decay
            age_days = (now - (item.updated_at or item.created_at)).total_seconds() / 86400.0
            recency = math.exp(-math.log(2) * age_days / max(half_life_days, 1e-6))

            final = 0.75 * sim + 0.25 * recency
            scored.append((item, final, sim, recency))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    def all(self, user_id: str, category: Optional[str] = None) -> List[MemoryItem]:
        return [x for x in self._items if x.user_id == user_id and (category is None or x.category == category)]

    def _get(self, item_id: str) -> MemoryItem:
        for x in self._items:
            if x.id == item_id:
                return x
        raise KeyError(f"Memory id not found: {item_id}")

def propose_memory_ops(user_text: str) -> List[Dict[str, Any]]:
    """
    Mock of an LLM deciding memory ops.
    Returns a list of ops like:
      {"op":"ADD", "category":"semantic", "key":"diet", "value":"vegetarian"}
    """
    t = user_text.lower()
    ops = []

    # semantic: dietary preference
    if "vegetarian" in t:
        ops.append({"op": "UPSERT", "category": "semantic", "key": "diet", "value": "vegetarian"})
    if "lactose" in t and ("intolerant" in t or "allergic" in t):
        ops.append({"op": "UPSERT", "category": "semantic", "key": "lactose", "value": "intolerant"})
    if "gluten" in t and ("allergic" in t or "intolerant" in t):
        ops.append({"op": "UPSERT", "category": "semantic", "key": "gluten", "value": "allergic"})

    # semantic: prefers working at night
    if "prefer working at night" in t or "i prefer working at night" in t:
        ops.append({"op": "UPSERT", "category": "semantic", "key": "work_pref", "value": "night"})

    # episodic: stress + deadline
    if "stressed" in t or "deadline" in t:
        ops.append({"op": "ADD_EPISODE", "category": "episodic"})

    # procedural: teach a procedure
    if "procedure:" in t or "steps:" in t:
        ops.append({"op": "ADD_PROCEDURE", "category": "procedure"})

    return ops


def apply_ops(store: MemoryStore, user_id: str, user_text: str, now: Optional[datetime] = None) -> None:
    now = now or datetime.utcnow()

    # We'll store semantic key-values as ONE memory per key:
    # metadata["kv_key"] = key
    # text is human readable
    ops = propose_memory_ops(user_text)

    for op in ops:
        if op["op"] == "UPSERT":
            key, value = op["key"], op["value"]

            # find existing semantic item for this key
            existing = [
                m for m in store.all(user_id, category="semantic")
                if m.metadata.get("kv_key") == key
            ]
            if existing:
                # UPDATE latest one (conflict resolution: overwrite)
                latest = max(existing, key=lambda m: (m.updated_at or m.created_at))
                store.update(
                    latest.id,
                    new_text=f"User {key} = {value}",
                    now=now,
                    kv_key=key,
                    kv_value=value,
                )
            else:
                store.add(
                    user_id=user_id,
                    category="semantic",
                    text=f"User {key} = {value}",
                    now=now,
                    kv_key=key,
                    kv_value=value,
                )

        elif op["op"] == "ADD_EPISODE":
            # Episodic summary: minimal “episode” derived from message
            store.add(
                user_id=user_id,
                category="episodic",
                text=f"Episode: {user_text.strip()}",
                now=now,
                summarized=False,
                source="chat",
            )

        elif op["op"] == "ADD_PROCEDURE":
            store.add(
                user_id=user_id,
                category="procedure",
                text=user_text.strip(),
                now=now,
                source="taught_by_user",
            )

def build_short_term_context(store: MemoryStore, user_id: str, user_query: str, now: Optional[datetime] = None) -> Dict[str, List[str]]:
    """
    Retrieves top memories and returns a structured context payload.
    """
    now = now or datetime.utcnow()

    sem = store.search(user_id, user_query, category="semantic", limit=5, now=now)
    epi = store.search(user_id, user_query, category="episodic", limit=3, now=now)
    proc = store.search(user_id, user_query, category="procedure", limit=2, now=now)

    context = {
        "semantic": [f"{m.text}" for (m, _, _, _) in sem],
        "episodic": [f"{m.text}" for (m, _, _, _) in epi],
        "procedure": [f"{m.text}" for (m, _, _, _) in proc],
    }
    return context


def agent_respond(user_query: str, context: Dict[str, List[str]]) -> str:
    """
    Mock response generator (in real life: call LLM with this context).
    Here we just demonstrate behavior.
    """
    sem = "\n".join(context["semantic"]) or "(none)"
    epi = "\n".join(context["episodic"]) or "(none)"
    proc = "\n".join(context["procedure"]) or "(none)"

    # Simple "decision": if question matches a procedure, suggest executing it
    if context["procedure"]:
        return (
            f"I found a relevant procedure you taught:\n\n{proc}\n\n"
            f"Based on your request: '{user_query}', I can follow those steps. "
            f"Do you want me to run it now (e.g., query DB / summarize / output format)?"
        )

    # If dietary constraints exist, respect them
    diet = [s for s in context["semantic"] if "diet" in s or "lactose" in s or "gluten" in s]
    if "recommend" in user_query.lower() and diet:
        return (
            f"Got it. I’ll respect your constraints:\n{chr(10).join(diet)}\n\n"
            f"Tell me your cuisine preference and budget, and I’ll suggest options."
        )

    return (
        f"Here’s what I remember that might matter:\n\n"
        f"SEMANTIC:\n{sem}\n\nEPISODIC:\n{epi}\n\n"
        f"Ask me what you want to do next based on this context."
    )

if __name__ == "__main__":
    store = MemoryStore()
    user_id = "thanh"

    base = datetime(2026, 2, 1, 12, 0, 0)

    # Day 1: user states preferences
    apply_ops(store, user_id, "I'm vegetarian and lactose intolerant.", now=base)

    # Day 2: user talks about stress & working style -> episodic + semantic
    apply_ops(store, user_id, "I'm stressed about my project deadline on Friday. I prefer working at night.", now=base + timedelta(days=1))

    # Day 3: user changes preference -> UPDATE semantic diet
    apply_ops(store, user_id, "Actually I'm not strictly vegetarian anymore. I eat fish sometimes.", now=base + timedelta(days=2))
    # (Our simple rules don't detect fish; below is manual upsert to demonstrate conflict update)
    apply_ops(store, user_id, "vegetarian", now=base + timedelta(days=2))  # keeps vegetarian; we’ll override manually:
    # override "diet"
    # Find diet memory and update
    diet_items = [m for m in store.all(user_id, "semantic") if m.metadata.get("kv_key") == "diet"]
    if diet_items:
        latest = max(diet_items, key=lambda m: (m.updated_at or m.created_at))
        store.update(latest.id, "User diet = pescatarian", now=base + timedelta(days=2), kv_key="diet", kv_value="pescatarian")

    # Day 4: user teaches a procedure
    apply_ops(
        store,
        user_id,
        "Procedure: monthly_report\nSteps:\n1. Query sales DB for last 30 days\n2. Summarize top 5 insights\n3. Ask output format",
        now=base + timedelta(days=3)
    )

    # Now: user asks something -> retrieve memory -> respond
    now = base + timedelta(days=10)
    query = "Can you recommend dinner? Also remind me what you know about my diet."
    context = build_short_term_context(store, user_id, query, now=now)
    print("=== SHORT-TERM CONTEXT LOADED ===")
    for k, v in context.items():
        print(f"\n[{k.upper()}]")
        for line in v:
            print(" -", line)

    print("\n=== AGENT RESPONSE ===")
    print(agent_respond(query, context))

    # Another query: procedure intent
    query2 = "How do we create the monthly report again?"
    context2 = build_short_term_context(store, user_id, query2, now=now)
    print("\n\n=== AGENT RESPONSE (PROCEDURE INTENT) ===")
    print(agent_respond(query2, context2))
