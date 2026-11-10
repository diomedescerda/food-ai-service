"""Versioned, identity-free contract from the trusted nutrition backend."""
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Metric = Literal["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]
EvidenceId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")]
Number = Annotated[float, Field(ge=0, le=1_000_000)]
Label = Annotated[str, Field(min_length=1, max_length=200)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PlanTarget(StrictModel):
    id: EvidenceId
    metric: Metric
    amount: Number
    unit: Literal["kcal", "g"]

    @model_validator(mode="after")
    def valid_unit(self):
        if self.unit != ("kcal" if self.metric == "calories" else "g"):
            raise ValueError("Target unit does not match metric")
        return self


class ApprovedPlan(StrictModel):
    id: EvidenceId
    version: Label
    targets: list[PlanTarget] = Field(min_length=1, max_length=5)


class CalculatedDiff(StrictModel):
    id: EvidenceId
    target_id: EvidenceId
    metric: Metric
    actual: Number | None
    target: Number
    delta: Annotated[float, Field(ge=-1_000_000, le=1_000_000)] | None
    unit: Literal["kcal", "g"]


class ConfirmedFood(StrictModel):
    id: EvidenceId
    name: Label
    grams: Annotated[float, Field(gt=0, le=100_000)] | None


class Restriction(StrictModel):
    id: EvidenceId
    label: Label


class MissingValue(StrictModel):
    id: EvidenceId
    field: Label
    label: Label


class ExplainRequest(StrictModel):
    schema_version: Literal["1"]
    request_id: str
    locale: Literal["es", "en"]
    plan: ApprovedPlan
    diffs: list[CalculatedDiff] = Field(max_length=5)
    confirmed_foods: list[ConfirmedFood] = Field(max_length=20)
    restrictions: list[Restriction] = Field(max_length=20)
    missing_values: list[MissingValue] = Field(max_length=30)

    @field_validator("request_id")
    @classmethod
    def valid_request_id(cls, value):
        UUID(value)
        return value

    @model_validator(mode="after")
    def validate_evidence(self):
        objects = [self.plan, *self.plan.targets, *self.diffs, *self.confirmed_foods, *self.restrictions, *self.missing_values]
        ids = [obj.id for obj in objects]
        if len(set(ids)) != len(ids):
            raise ValueError("Evidence IDs must be unique")
        targets = {target.id: target for target in self.plan.targets}
        if len({target.metric for target in self.plan.targets}) != len(targets):
            raise ValueError("Only one target per metric is allowed")
        if len({diff.target_id for diff in self.diffs}) != len(self.diffs):
            raise ValueError("Only one difference per target is allowed")
        for diff in self.diffs:
            target = targets.get(diff.target_id)
            if target is None or (diff.metric, diff.target, diff.unit) != (target.metric, target.amount, target.unit):
                raise ValueError("Difference must reference an unchanged approved target")
            if (diff.actual is None) != (diff.delta is None):
                raise ValueError("Unknown totals require an unknown difference")
            if diff.actual is not None and abs(diff.actual - diff.target - diff.delta) > 0.01:
                raise ValueError("Difference does not match the server calculated values")
        return self


class ExplanationItem(StrictModel):
    id: str
    claim: Literal["metric_diff", "confirm_portion", "missing_value", "review_restriction", "review_plan"]
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=3)
    text: str = Field(min_length=1, max_length=1000)


class ExplanationProvenance(StrictModel):
    provider: Literal["deterministic", "llm"]
    vendor: Literal["local", "anthropic", "openai"]
    model: str
    schema_version: Literal["1"] = "1"
    prompt_version: Literal["foodai-explain-v1"] = "foodai-explain-v1"
    fallback_used: bool
    fallback_reason: str | None
    attempted_vendors: list[str]


class ExplainResponse(StrictModel):
    schema_version: Literal["1"] = "1"
    request_id: str
    locale: Literal["es", "en"]
    observations: list[ExplanationItem] = Field(max_length=5)
    questions: list[ExplanationItem] = Field(max_length=70)
    suggestions: list[ExplanationItem] = Field(max_length=5)
    provenance: ExplanationProvenance
