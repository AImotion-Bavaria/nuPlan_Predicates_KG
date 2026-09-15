from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class AssertionKind(str, Enum):
    NATIVE = "native"
    DERIVED = "derived"


class ValueType(str, Enum):
    ENTITY = "entity"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DATETIME = "datetime"
    VECTOR2 = "vector2"


class PredicateDefinition(BaseModel):
    predicate_id: str
    label: str
    category: str
    description: str
    subject_types: List[str]
    object_types: List[str]
    value_type: ValueType = ValueType.ENTITY
    assertion_kind: AssertionKind
    source_fields: List[str] = Field(default_factory=list)
    rule_id: Optional[str] = None
    inverse_of: Optional[str] = None
    symmetric: bool = False
    temporal_scope: str = "instant"
    units: Optional[str] = None

    @model_validator(mode="after")
    def derived_requires_rule(self) -> "PredicateDefinition":
        if self.assertion_kind == AssertionKind.DERIVED and not self.rule_id:
            raise ValueError("Derived predicates require rule_id")
        return self


class RuleDefinition(BaseModel):
    rule_id: str
    version: str = "1.0.0"
    name: str
    description: str
    deterministic: bool = True
    inputs: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expression: str
    implementation: str
    references: List[str] = Field(default_factory=list)


class Provenance(BaseModel):
    dataset: str = "nuPlan"
    dataset_version: str = "v1.1"
    split: str
    log_name: str
    scenario_token: str
    timestamp_us: int
    source_table: Optional[str] = None
    source_fields: List[str] = Field(default_factory=list)
    rule_id: Optional[str] = None
    rule_version: Optional[str] = None
    extractor_version: str = "1.0.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PredicateAssertion(BaseModel):
    assertion_id: str
    predicate_id: str
    subject_id: str
    object_id: Optional[str] = None
    value: Optional[Any] = None
    value_type: ValueType
    valid_time_us: int
    end_time_us: Optional[int] = None
    assertion_kind: AssertionKind
    provenance: Provenance
    evidence: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def object_xor_value(self) -> "PredicateAssertion":
        if (self.object_id is None) == (self.value is None):
            raise ValueError("Exactly one of object_id or value must be present")
        if self.assertion_kind == AssertionKind.DERIVED and not self.provenance.rule_id:
            raise ValueError("Derived assertions require provenance.rule_id")
        return self
