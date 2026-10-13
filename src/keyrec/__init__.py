"""Bounded-memory KeyRec core and model adapters.

The algorithmic memory and readout modules are model-independent. Narrow model
adapters own architecture-specific extraction and visual-token injection.
"""

from .encoder import (
    KeyRecMemoryWriter,
    MemoryWriteConfig,
    build_event_candidate,
    novelty_against_routes,
)
from .gemma import GemmaKeyRecAdapter, PreparedGemmaInput
from .memory import (
    BoundedEventBank,
    EventCandidate,
    EventMemoryConfig,
    MemoryUpdate,
    RecentCache,
    StoredEvent,
    TokenPayload,
)
from .neo import (
    NeoKeyRecAdapter,
    NeoVisualSpan,
    PreparedNeoInput,
    build_neo_visual_spans,
)
from .readout import (
    MemorySegment,
    QueryAdaptiveReadout,
    ReadoutConfig,
    ReadoutResult,
    RouterDecision,
    RouterState,
    build_router_prompt,
    parse_router_output,
)
from .pipeline import (
    KeyRecModelAdapter,
    KeyRecPipeline,
    PipelineConfig,
    PipelineResult,
)

__all__ = [
    "BoundedEventBank",
    "EventCandidate",
    "EventMemoryConfig",
    "GemmaKeyRecAdapter",
    "KeyRecMemoryWriter",
    "KeyRecModelAdapter",
    "KeyRecPipeline",
    "MemorySegment",
    "MemoryUpdate",
    "MemoryWriteConfig",
    "NeoKeyRecAdapter",
    "NeoVisualSpan",
    "PipelineConfig",
    "PipelineResult",
    "PreparedGemmaInput",
    "PreparedNeoInput",
    "QueryAdaptiveReadout",
    "ReadoutConfig",
    "ReadoutResult",
    "RecentCache",
    "RouterDecision",
    "RouterState",
    "StoredEvent",
    "TokenPayload",
    "build_event_candidate",
    "build_neo_visual_spans",
    "build_router_prompt",
    "novelty_against_routes",
    "parse_router_output",
]
