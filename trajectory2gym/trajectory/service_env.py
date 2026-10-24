from __future__ import annotations

import copy
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from .loader import adapter_for_source, load_trajectory_episodes
from .recorded_state_graph import RecordedStateGraph
from .schema import TrajectoryEpisode
from .session import TrajectoryGymSession
from .verifiability import VerifiabilityAnalyzer
from .tool_semantics import ToolSemantics, load_tool_semantics_file, tool_semantics_from_mapping


class Trajectory2GymServiceEnv:
    """Single service wrapper for trajectory-backed training environments."""

    def __init__(
        self,
        task_id: str | None = None,
        instance_id: str | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.task_id = task_id
        self.instance_id = instance_id
        self.params: dict[str, Any] = {}
        self.data_path: Path | None = None
        self.source = "auto"
        self.language = "en"
        self.use_conversation_history = True
        self.max_entries: int | None = None
        self.max_tasks: int | None = None
        self.task_ids: frozenset[str] | None = None
        self.observation_db_task_ids: frozenset[str] | None = None
        self.verifiability_mode = True
        self._all_episodes: list[TrajectoryEpisode] = []
        self.episodes: list[TrajectoryEpisode] = []
        self.episode: TrajectoryEpisode | None = None
        self.session: TrajectoryGymSession | None = None
        self._configure(params or {})

    @staticmethod
    def get_query_list(split: str = "train", params: dict[str, Any] | None = None) -> list[str]:
        del split
        env = Trajectory2GymServiceEnv(params=params or {})
        return [episode.id for episode in env._load_episodes()]

    def get_init_state(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params:
            self._configure(params)
        # Keep the selectable episode pool separate from the full evidence
        # pool.  A heldout task allow-list must not make train evidence
        # disappear before the observation DB is built.
        self._all_episodes = self._load_all_episodes()
        self.episodes = self._selectable_episodes(self._all_episodes)
        self.episode = self._select_episode(self.task_id)
        evidence_episodes = self._observation_db_episodes()
        adapter = adapter_for_source(self.episode.source)
        adapter.tool_semantics = _tool_semantics_from_params(
            self.params, default=adapter.tool_semantics
        )
        accept_equivalent = _truthy(self.params.get("accept_equivalent_replay", True))
        adapter.accept_equivalent = accept_equivalent
        evidence_ids = tuple(sorted(episode.id for episode in evidence_episodes))
        if self.verifiability_mode:
            adapter.tool_semantics, analyzer = _cached_verifier_bundle(
                self._cache_key(),
                evidence_ids,
                accept_equivalent,
                adapter.tool_semantics,
            )
        else:
            adapter.tool_semantics = compiled_tool_semantics(
                evidence_episodes, default=adapter.tool_semantics
            )
            analyzer = None
        graph_enabled = bool(
            analyzer is not None
            and _truthy(
                self.params.get(
                    "recorded_state_graph",
                    self.params.get("trajectory_branching", False),
                )
            )
        )
        recorded_state_graph = (
            _cached_recorded_state_graph(self._cache_key(), evidence_ids)
            if graph_enabled
            else None
        )
        candidate_episodes = (
            _trajectory_candidates(self.episode, evidence_episodes)
            if analyzer is not None
            # Switching to another recorded trajectory with the same visible
            # prefix is an evidence-backed extension, but it is not part of
            # the normative five-way Partial contract.  Core Partial keeps a
            # DB hit non-advancing, so trajectory-trie progress must be an
            # explicit opt-in experiment.
            and _truthy(self.params.get("trajectory_branching", False))
            else ()
        )
        self.session = TrajectoryGymSession(
            self.episode,
            candidate_episodes=candidate_episodes,
            adapter=adapter,
            verifiability_analyzer=analyzer,
            recorded_state_graph=recorded_state_graph,
            terminate_on_rejected_side_effect=_truthy(
                self.params.get("terminate_on_rejected_side_effect", False)
            ),
            max_rejected_attempts_per_step=_optional_int(
                self.params.get("max_rejected_attempts_per_step")
            ),
        )
        result = self.session.reset()
        result["info"]["tools"] = copy.deepcopy(list(self.episode.tools))
        return result

    def _observation_db_episodes(self) -> list[TrajectoryEpisode]:
        """Keep selectable validation episodes separate from DB evidence."""

        if self.observation_db_task_ids is None:
            return self.episodes
        return [episode for episode in self._all_episodes if episode.id in self.observation_db_task_ids]

    def step(self, action: dict[str, Any] | list[dict[str, Any]] | str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        del params
        session = self._require_session()
        result = session.step(_normalize_action(action))
        result["info"]["tools"] = copy.deepcopy(list(session.episode.tools))
        return result

    def evaluate(self, messages: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> Any:
        del messages
        eval_params = {**self.params, **(params or {})}
        sparse = bool(eval_params.get("sparse", True))
        return self._require_session().evaluate(sparse=sparse, params=eval_params)

    def get_info(
        self,
        messages: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del messages, params
        session = self._require_session()
        info = session.info()
        info["env_type"] = "trajectory2gym"
        info["tools"] = copy.deepcopy(list(session.episode.tools))
        return info

    def close(self) -> None:
        self.session = None

    def _configure(self, params: dict[str, Any]) -> None:
        self.params.update(params)
        data_path = self.params.get("data_path")
        self.data_path = Path(data_path) if data_path else None
        self.source = str(self.params.get("source") or "auto")
        self.language = str(self.params.get("language") or "en")
        self.use_conversation_history = bool(self.params.get("use_conversation_history", True))
        self.max_entries = _optional_int(self.params.get("max_entries"))
        self.max_tasks = _optional_int(self.params.get("max_tasks"))
        self.task_ids = _task_ids_from_params(self.params)
        self.observation_db_task_ids = _task_ids_from_params(
            {
                "task_ids": self.params.get("observation_db_task_ids"),
                "task_ids_file": self.params.get("observation_db_task_ids_file"),
            }
        )
        verifier_setting = self.params.get(
            "verifiability_mode", self.params.get("partial_environment", True)
        )
        self.verifiability_mode = _truthy(verifier_setting)

    def _load_episodes(self) -> list[TrajectoryEpisode]:
        """Load the selectable episode pool (honoring ``task_ids``)."""

        return self._selectable_episodes(self._load_all_episodes())

    def _load_all_episodes(self) -> list[TrajectoryEpisode]:
        if self.data_path is None:
            raise ValueError("trajectory2gym requires params['data_path']")
        path = self.data_path.resolve()
        stat = path.stat()
        episodes = list(
            _cached_episodes(
                str(path),
                stat.st_mtime_ns,
                stat.st_size,
                self.source,
                self.language,
                self.use_conversation_history,
                self.max_entries,
                self.max_tasks,
            )
        )
        return episodes

    def _cache_key(self) -> tuple[Any, ...]:
        if self.data_path is None:
            raise ValueError("trajectory2gym requires params['data_path']")
        path = self.data_path.resolve()
        stat = path.stat()
        return (
            str(path),
            stat.st_mtime_ns,
            stat.st_size,
            self.source,
            self.language,
            self.use_conversation_history,
            self.max_entries,
            self.max_tasks,
        )

    def _selectable_episodes(self, episodes: list[TrajectoryEpisode]) -> list[TrajectoryEpisode]:
        if self.task_ids is None:
            return list(episodes)
        return [episode for episode in episodes if episode.id in self.task_ids]

    def _select_episode(self, task_id: str | None) -> TrajectoryEpisode:
        if not self.episodes:
            raise ValueError(f"no trajectory episodes loaded from {self.data_path}")
        if task_id is None:
            return self.episodes[0]
        for episode in self.episodes:
            if episode.id == task_id:
                return episode
        raise KeyError(f"unknown trajectory2gym task_id: {task_id}")

    def _require_session(self) -> TrajectoryGymSession:
        if self.session is None:
            raise RuntimeError("environment is not initialized; call get_init_state first")
        return self.session


def _normalize_action(action: dict[str, Any] | list[dict[str, Any]] | str) -> dict[str, Any] | str:
    if isinstance(action, list):
        return copy.deepcopy(action[-1]) if action else {"role": "assistant", "content": ""}
    if isinstance(action, dict) and "name" in action and "role" not in action:
        return {
            "role": "assistant",
            "content": f"<tool_call>{json.dumps(action, ensure_ascii=False, separators=(',', ':'))}</tool_call>",
        }
    return copy.deepcopy(action)


def _trajectory_candidates(
    episode: TrajectoryEpisode,
    evidence_episodes: list[TrajectoryEpisode],
) -> tuple[TrajectoryEpisode, ...]:
    """Return recorded continuations of one explicitly declared state.

    A task/group identifier is not a state identifier: decision windows from
    different points of the same task may have different backend state and
    visible evidence.  The compiler must therefore declare a
    ``trajectory_state_id`` only after checking its context/action/observation
    ledger.  Core runtime never derives state equivalence from benchmark task
    ids or natural-language similarity.
    """

    state_id = _trajectory_state_id(episode)
    if not state_id:
        return (episode,)
    return tuple(
        candidate
        for candidate in evidence_episodes
        if _trajectory_state_id(candidate) == state_id
        and candidate.source == episode.source
        and candidate.domain == episode.domain
        and candidate.tools == episode.tools
        # Failed sessions remain valid observation evidence, but they must
        # never become a positive policy continuation merely because their
        # recorded edge is reachable from the same state node.
        and candidate.metadata.get("training_eligible", True) is True
    )


def _trajectory_state_id(episode: TrajectoryEpisode) -> str:
    return str(episode.metadata.get("trajectory_state_id") or "").strip()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _task_ids_from_params(params: dict[str, Any]) -> frozenset[str] | None:
    """Resolve an optional episode allow-list without coupling the runtime to a benchmark.

    A newline-delimited file is the simple interchange format.  JSON manifests
    may also expose ``train_task_ids``/``val_task_ids``/``task_ids``/``ids`` so
    split manifests can be passed directly by launchers.
    """

    raw = params.get("task_ids")
    task_ids_file = params.get("task_ids_file")
    if raw is None and task_ids_file:
        path = Path(str(task_ids_file))
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
        if isinstance(payload, dict):
            for key in ("train_task_ids", "val_task_ids", "task_ids", "ids"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    raw = candidate
                    break
        elif payload is None:
            raw = path.read_text(encoding="utf-8").splitlines()
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.replace(",", "\n").splitlines()
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError("task_ids must be a sequence, comma/newline-delimited string, or task_ids_file")
    resolved = frozenset(str(item).strip() for item in raw if str(item).strip())
    return resolved


def compiled_tool_semantics(
    episodes: list[TrajectoryEpisode] | tuple[TrajectoryEpisode, ...],
    *,
    default: ToolSemantics,
) -> ToolSemantics:
    """Compile a thin read/write declaration from trajectory metadata.

    Explicit episode-level safe-tool declarations and per-action
    ``observation_db_safe`` booleans use one generic format; ``mutates_state``
    is an independent write veto. If no safe declaration is present,
    exact/equivalent replay remains available but cross-trajectory Observation
    DB reuse stays disabled. Tool-name prefixes do not prove context safety.
    """

    declared_safe_tools: set[str] = set(
        default.observation_db_safe_exact_names or ()
    )
    mutation_values: dict[str, set[bool]] = {}
    db_safety_values: dict[str, set[bool]] = {}
    saw_declaration = default.observation_db_safe_exact_names is not None
    for episode in episodes:
        raw_safe_tools = episode.metadata.get("observation_db_safe_tools")
        if isinstance(raw_safe_tools, list):
            saw_declaration = True
            declared_safe_tools.update(str(name) for name in raw_safe_tools if str(name))
        for step in episode.steps:
            raw_metadata = step.metadata.get("action_metadata")
            if not isinstance(raw_metadata, list):
                continue
            for index, action in enumerate(step.expected_actions):
                if index >= len(raw_metadata) or not isinstance(raw_metadata[index], dict):
                    continue
                mutates_state = raw_metadata[index].get("mutates_state")
                if isinstance(mutates_state, bool):
                    saw_declaration = True
                    mutation_values.setdefault(action.name, set()).add(mutates_state)
                observation_db_safe = raw_metadata[index].get(
                    "observation_db_safe"
                )
                if isinstance(observation_db_safe, bool):
                    saw_declaration = True
                    db_safety_values.setdefault(action.name, set()).add(
                        observation_db_safe
                    )
    if not saw_declaration:
        return default
    declared_safe_tools.update(
        name for name, values in db_safety_values.items() if values == {True}
    )
    # A write declaration always wins over an episode-level safe declaration.
    # This keeps contradictory or mixed evidence out of cross-trajectory replay.
    declared_safe_tools.difference_update(
        name for name, values in mutation_values.items() if True in values
    )
    declared_safe_tools.difference_update(
        name for name, values in db_safety_values.items() if False in values
    )
    declared_mutating_tools = {
        ToolSemantics.normalize(name)
        for name, values in mutation_values.items()
        if True in values
    }
    return ToolSemantics(
        side_effect_exact_names=tuple(
            sorted(
                {
                    ToolSemantics.normalize(name)
                    for name in default.side_effect_exact_names
                }
                | declared_mutating_tools
            )
        ),
        observation_db_safe_exact_names=tuple(
            sorted({ToolSemantics.normalize(name) for name in declared_safe_tools if str(name)})
        ),
        allow_name_inferred_observation_db_replay=False,
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "on"}


def _tool_semantics_from_params(
    params: dict[str, Any], *, default: ToolSemantics
) -> ToolSemantics:
    """Load the thin, dataset-owned semantics declaration if provided.

    The file is plain JSON and contains names only; matching, state safety,
    verifier behavior, and rewards remain in the shared core.
    """

    value: Any = params.get("tool_semantics")
    path_value = params.get("tool_semantics_file")
    if value is None and path_value:
        return load_tool_semantics_file(str(path_value), default=default)
    if value is None:
        return default
    if not isinstance(value, dict):
        raise ValueError("tool_semantics must be a JSON object or tool_semantics_file")

    return tool_semantics_from_mapping(value, default=default)


_VERIFIER_COMPILE_LOCK = threading.Lock()


@lru_cache(maxsize=8)
def _cached_verifier_bundle(
    episode_cache_key: tuple[Any, ...],
    evidence_ids: tuple[str, ...],
    accept_equivalent: bool,
    default_semantics: ToolSemantics,
) -> tuple[ToolSemantics, VerifiabilityAnalyzer]:
    """Compile immutable evidence once per service process.

    Environment instances share neither cursor nor reward state; only the
    read-only observation index and declarative tool semantics are reused.
    The file stat, parser settings, evidence IDs and verifier settings are all
    part of the cache key, so changed data cannot reuse stale evidence.
    """

    # ``functools.lru_cache`` may execute the same miss concurrently. Route
    # misses through a serialized inner cache so a burst of create-instance
    # requests cannot compile the same large observation DB dozens of times.
    with _VERIFIER_COMPILE_LOCK:
        return _serialized_verifier_bundle(
            episode_cache_key,
            evidence_ids,
            accept_equivalent,
            default_semantics,
        )


@lru_cache(maxsize=8)
def _cached_recorded_state_graph(
    episode_cache_key: tuple[Any, ...],
    evidence_ids: tuple[str, ...],
) -> RecordedStateGraph:
    """Compile immutable explicit state edges once per evidence pool."""

    with _VERIFIER_COMPILE_LOCK:
        return _serialized_recorded_state_graph(
            episode_cache_key, evidence_ids
        )


@lru_cache(maxsize=8)
def _serialized_recorded_state_graph(
    episode_cache_key: tuple[Any, ...],
    evidence_ids: tuple[str, ...],
) -> RecordedStateGraph:
    """Serialized inner cache avoids duplicate graph builds on request bursts."""

    episodes = _cached_episodes(*episode_cache_key)
    allowed = frozenset(evidence_ids)
    return RecordedStateGraph.from_episodes(
        episode for episode in episodes if episode.id in allowed
    )


@lru_cache(maxsize=8)
def _serialized_verifier_bundle(
    episode_cache_key: tuple[Any, ...],
    evidence_ids: tuple[str, ...],
    accept_equivalent: bool,
    default_semantics: ToolSemantics,
) -> tuple[ToolSemantics, VerifiabilityAnalyzer]:
    episodes = _cached_episodes(*episode_cache_key)
    allowed = frozenset(evidence_ids)
    evidence_episodes = tuple(
        episode for episode in episodes if episode.id in allowed
    )
    semantics = compiled_tool_semantics(
        evidence_episodes, default=default_semantics
    )
    analyzer = VerifiabilityAnalyzer.from_episodes(
        evidence_episodes,
        accept_equivalent=accept_equivalent,
        tool_semantics=semantics,
    )
    return semantics, analyzer


@lru_cache(maxsize=4)
def _cached_episodes(
    path: str,
    mtime_ns: int,
    size: int,
    source: str,
    language: str,
    use_conversation_history: bool,
    max_entries: int | None,
    max_tasks: int | None,
) -> tuple[TrajectoryEpisode, ...]:
    del mtime_ns, size
    return tuple(
        load_trajectory_episodes(
            path,
            source=source,
            language=language,
            use_conversation_history=use_conversation_history,
            max_entries=max_entries,
            max_tasks=max_tasks,
        )
    )
