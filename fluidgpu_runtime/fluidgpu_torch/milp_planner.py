from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fluidgpu_torch.planner import CsvTask, PlanConfig, PlanState


ConstraintSense = Literal["<=", ">=", "=="]


@dataclass(frozen=True)
class LinearTerm:
    variable: str
    coefficient: float


@dataclass(frozen=True)
class LinearConstraint:
    name: str
    terms: tuple[LinearTerm, ...]
    sense: ConstraintSense
    rhs: float


@dataclass(frozen=True)
class LinearObjective:
    terms: tuple[LinearTerm, ...]
    constant: float
    sense: Literal["min"] = "min"


@dataclass(frozen=True)
class MILPModel:
    binary_variables: tuple[str, ...]
    constraints: tuple[LinearConstraint, ...]
    objective: LinearObjective
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MILPSolution:
    plan: PlanState
    solver: str
    gurobi_status: str


def build_two_rank_milp(
    *,
    tasks: list[str],
    rank0: dict[str, CsvTask],
    rank1: dict[str, CsvTask],
    config: PlanConfig,
) -> MILPModel:
    assert "embed" in rank0 and "norm_lm_head" in rank0, "rank0 CSV must include fixed tasks"
    assert "embed" in rank1 and "norm_lm_head" in rank1, "rank1 CSV must include fixed tasks"
    task_vars = tuple(task_variable(name) for name in tasks)
    switch_vars = tuple(switch_variable(index) for index in range(len(tasks) + 1))
    binary_variables = (*task_vars, *switch_vars)
    constraints: list[LinearConstraint] = []
    for index in range(len(tasks) + 1):
        left = None if index == 0 else task_vars[index - 1]
        right = None if index == len(tasks) else task_vars[index]
        constraints.extend(switch_constraints(index, left=left, right=right))

    objective_terms: list[LinearTerm] = []
    objective_constant = fixed_task_cost(rank0["embed"], config) + fixed_task_cost(
        rank0["norm_lm_head"],
        config,
    )
    for task_name, variable in zip(tasks, task_vars):
        rank0_cost = fixed_task_cost(rank0[task_name], config)
        rank1_cost = fixed_task_cost(rank1[task_name], config)
        objective_constant += rank0_cost
        objective_terms.append(LinearTerm(variable, rank1_cost - rank0_cost))

    switch_cost = config.prefill_transfer_us + config.decode_weight * config.decode_transfer_us
    for variable in switch_vars:
        objective_terms.append(LinearTerm(variable, switch_cost))

    return MILPModel(
        binary_variables=binary_variables,
        constraints=tuple(constraints),
        objective=LinearObjective(tuple(objective_terms), objective_constant),
        metadata={
            "rank_count": 2,
            "task_count": len(tasks),
            "start_rank": 0,
            "end_rank": 0,
            "prompt_len": config.prompt_len,
            "comm_bw_gbps": config.comm_bw_gbps,
            "decode_weight": config.decode_weight,
            "prefill_transfer_us": config.prefill_transfer_us,
            "decode_transfer_us": config.decode_transfer_us,
            "runtime_compatible": False,
            "note": "Standalone MILP formulation only; not connected to fluidgpu_torch runtime.",
        },
    )


def build_multi_rank_milp(
    *,
    tasks: list[str],
    rank_costs: list[dict[str, CsvTask]],
    config: PlanConfig,
) -> MILPModel:
    rank_count = len(rank_costs)
    assert rank_count >= 2, f"rank_count must be >= 2, got {rank_count}"
    for rank, costs in enumerate(rank_costs):
        assert "embed" in costs and "norm_lm_head" in costs, (
            f"rank{rank} CSV must include fixed tasks"
        )

    task_vars = tuple(
        task_rank_variable(task_name, rank)
        for task_name in tasks
        for rank in range(rank_count)
    )
    boundary_pair_vars = tuple(
        boundary_pair_variable(index, left_rank, right_rank)
        for index in range(len(tasks) + 1)
        for left_rank in range(rank_count)
        for right_rank in range(rank_count)
    )
    binary_variables = (*task_vars, *boundary_pair_vars)

    constraints: list[LinearConstraint] = []
    constraints.extend(
        assignment_constraint(task_name, rank_count=rank_count)
        for task_name in tasks
    )
    for index in range(len(tasks) + 1):
        constraints.extend(
            boundary_pair_constraints(
                index,
                tasks=tasks,
                rank_count=rank_count,
            )
        )

    objective_terms: list[LinearTerm] = []
    objective_constant = fixed_task_cost(rank_costs[0]["embed"], config) + fixed_task_cost(
        rank_costs[0]["norm_lm_head"],
        config,
    )
    for task_name in tasks:
        for rank, costs in enumerate(rank_costs):
            assert task_name in costs, f"rank{rank} CSV missing task {task_name}"
            objective_terms.append(
                LinearTerm(
                    task_rank_variable(task_name, rank),
                    fixed_task_cost(costs[task_name], config),
                )
            )

    switch_cost = config.prefill_transfer_us + config.decode_weight * config.decode_transfer_us
    for index in range(len(tasks) + 1):
        for left_rank in range(rank_count):
            for right_rank in range(rank_count):
                if left_rank == right_rank:
                    continue
                objective_terms.append(
                    LinearTerm(
                        boundary_pair_variable(index, left_rank, right_rank),
                        switch_cost,
                    )
                )

    return MILPModel(
        binary_variables=binary_variables,
        constraints=tuple(constraints),
        objective=LinearObjective(tuple(objective_terms), objective_constant),
        metadata={
            "rank_count": rank_count,
            "task_count": len(tasks),
            "start_rank": 0,
            "end_rank": 0,
            "prompt_len": config.prompt_len,
            "comm_bw_gbps": config.comm_bw_gbps,
            "decode_weight": config.decode_weight,
            "prefill_transfer_us": config.prefill_transfer_us,
            "decode_transfer_us": config.decode_transfer_us,
            "runtime_compatible": False,
            "note": "Standalone multi-rank MILP formulation only; not connected to fluidgpu_torch runtime.",
        },
    )


def solve_with_gurobi(
    *,
    tasks: list[str],
    rank0: dict[str, CsvTask],
    rank1: dict[str, CsvTask],
    config: PlanConfig,
) -> MILPSolution:
    return solve_multi_rank_with_gurobi(
        tasks=tasks,
        rank_costs=[rank0, rank1],
        config=config,
    )


def solve_multi_rank_with_gurobi(
    *,
    tasks: list[str],
    rank_costs: list[dict[str, CsvTask]],
    config: PlanConfig,
) -> MILPSolution:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "gurobipy is required for solve_with_gurobi. "
            "Install via `pip install gurobipy` and configure a Gurobi license."
        ) from exc

    rank_count = len(rank_costs)
    milp_model = build_multi_rank_milp(tasks=tasks, rank_costs=rank_costs, config=config)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        m = gp.Model("fluidgpu_milp", env=env)
        gvars: dict[str, gp.Var] = {
            name: m.addVar(vtype=GRB.BINARY, name=name)
            for name in milp_model.binary_variables
        }
        for constraint in milp_model.constraints:
            expr = gp.quicksum(
                term.coefficient * gvars[term.variable] for term in constraint.terms
            )
            if constraint.sense == "<=":
                m.addConstr(expr <= constraint.rhs, name=constraint.name)
            elif constraint.sense == ">=":
                m.addConstr(expr >= constraint.rhs, name=constraint.name)
            else:
                m.addConstr(expr == constraint.rhs, name=constraint.name)
        obj_expr = gp.quicksum(
            term.coefficient * gvars[term.variable] for term in milp_model.objective.terms
        )
        m.setObjective(obj_expr + milp_model.objective.constant, GRB.MINIMIZE)
        m.optimize()
        status = _gurobi_status_name(m.status)
        if m.status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Gurobi did not return OPTIMAL for fluidgpu MILP; status={status}. "
                "License size limits or infeasibility may apply."
            )
        ranks = tuple(
            selected_rank_for_task(
                task_name=name,
                rank_count=rank_count,
                values={
                    rank: float(gvars[task_rank_variable(name, rank)].X)
                    for rank in range(rank_count)
                },
            )
            for name in tasks
        )
    finally:
        env.dispose()

    plan = evaluate_assignment_multi(
        tasks=tasks,
        ranks=ranks,
        rank_costs=rank_costs,
        config=config,
    )
    return MILPSolution(plan=plan, solver="gurobi", gurobi_status=status)


@dataclass(frozen=True)
class PerPhaseSolution:
    prefill_ranks: tuple[int, ...]
    decode_ranks: tuple[int, ...]
    objective_us: float
    prefill_us: float
    decode_us: float
    solver: str
    gurobi_status: str


def _is_phase_mobile(task_name: str) -> bool:
    """Mirror of the runtime's decode_rank contract (profile.py): only the
    coarse whole-FFN group ("..._mlp") may decode on a different rank than it
    prefills on. Attention groups are KV-bound, and every fine-grained
    sub-group (gate_up/down_proj, moe router/experts/combine) exchanges
    intra-group aux tensors whose peers are resolved phase-agnostically —
    the profile loader rejects a decode_rank on any of them, so the planner
    must not emit one."""
    return task_name.endswith("_mlp")


def solve_per_phase_with_gurobi(
    *,
    tasks: list[str],
    rank0: dict[str, CsvTask],
    rank1: dict[str, CsvTask],
    config: PlanConfig,
) -> PerPhaseSolution:
    """Joint prefill/decode placement: stateless FFN groups may move between
    phases; attention groups (KV owners) keep one rank across both."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError("gurobipy is required for per-phase planning") from exc

    sources = (rank0, rank1)
    phases = ("prefill", "decode")
    phase_cost = {
        "prefill": lambda task, rank: sources[rank][task].prefill_us,
        "decode": lambda task, rank: sources[rank][task].decode_us,
    }
    switch_cost = {
        "prefill": config.prefill_transfer_us,
        "decode": config.decode_transfer_us,
    }
    phase_weight = {"prefill": 1.0, "decode": config.decode_weight}

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        m = gp.Model("fluidgpu_per_phase", env=env)
        x = {
            (t, r, p): m.addVar(vtype=GRB.BINARY, name=f"x_{t}_r{r}_{p}")
            for t in tasks
            for r in (0, 1)
            for p in phases
        }
        for t in tasks:
            for p in phases:
                m.addConstr(x[(t, 0, p)] + x[(t, 1, p)] == 1)
            if not _is_phase_mobile(t):
                for r in (0, 1):
                    m.addConstr(x[(t, r, "prefill")] == x[(t, r, "decode")])

        objective = gp.LinExpr()
        for p in phases:
            w = phase_weight[p]
            chain = ["embed", *tasks, "norm_lm_head"]
            for t in tasks:
                for r in (0, 1):
                    objective += w * phase_cost[p](t, r) * x[(t, r, p)]
            objective += w * (phase_cost[p]("embed", 0) + phase_cost[p]("norm_lm_head", 0))
            # boundary switches: |x_left - x_right| on rank0 indicator
            for left, right in zip(chain, chain[1:]):
                lvar = 1.0 if left in ("embed", "norm_lm_head") else None
                rvar = 1.0 if right in ("embed", "norm_lm_head") else None
                s = m.addVar(vtype=GRB.BINARY, name=f"s_{left}__{right}_{p}")
                lexpr = lvar if lvar is not None else x[(left, 0, p)]
                rexpr = rvar if rvar is not None else x[(right, 0, p)]
                m.addConstr(s >= lexpr - rexpr)
                m.addConstr(s >= rexpr - lexpr)
                objective += w * switch_cost[p] * s
        m.setObjective(objective, GRB.MINIMIZE)
        m.optimize()
        status = _gurobi_status_name(m.status)
        assert m.status == GRB.OPTIMAL, f"per-phase MILP not optimal: {status}"

        def picked(t: str, p: str) -> int:
            return 0 if x[(t, 0, p)].X > 0.5 else 1

        prefill_ranks = tuple(picked(t, "prefill") for t in tasks)
        decode_ranks = tuple(picked(t, "decode") for t in tasks)

        def phase_total(p: str, ranks: tuple[int, ...]) -> float:
            total = phase_cost[p]("embed", 0) + phase_cost[p]("norm_lm_head", 0)
            prev = 0
            for t, r in zip(tasks, ranks):
                if r != prev:
                    total += switch_cost[p]
                total += phase_cost[p](t, r)
                prev = r
            if prev != 0:
                total += switch_cost[p]
            return total

        prefill_us = phase_total("prefill", prefill_ranks)
        decode_us = phase_total("decode", decode_ranks)
    finally:
        env.dispose()

    return PerPhaseSolution(
        prefill_ranks=prefill_ranks,
        decode_ranks=decode_ranks,
        objective_us=prefill_us + config.decode_weight * decode_us,
        prefill_us=prefill_us,
        decode_us=decode_us,
        solver="gurobi",
        gurobi_status=status,
    )


_GUROBI_STATUS_NAMES = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
}


def _gurobi_status_name(status: int) -> str:
    return _GUROBI_STATUS_NAMES.get(status, f"UNKNOWN_{status}")


def evaluate_assignment(
    *,
    tasks: list[str],
    ranks: tuple[int, ...],
    rank0: dict[str, CsvTask],
    rank1: dict[str, CsvTask],
    config: PlanConfig,
) -> PlanState:
    return evaluate_assignment_multi(
        tasks=tasks,
        ranks=ranks,
        rank_costs=[rank0, rank1],
        config=config,
    )


def evaluate_assignment_multi(
    *,
    tasks: list[str],
    ranks: tuple[int, ...],
    rank_costs: list[dict[str, CsvTask]],
    config: PlanConfig,
) -> PlanState:
    assert len(tasks) == len(ranks), "tasks and ranks must have the same length"
    assert len(rank_costs) >= 2, "rank_costs must contain at least two ranks"
    sources = tuple(rank_costs)
    prefill = rank_costs[0]["embed"].prefill_us
    decode = rank_costs[0]["embed"].decode_us
    prev_rank = 0
    assignments: list[tuple[str, int]] = []
    for task_name, rank in zip(tasks, ranks):
        assert 0 <= rank < len(sources), f"invalid rank {rank} for {task_name}"
        if rank != prev_rank:
            prefill += config.prefill_transfer_us
            decode += config.decode_transfer_us
        prefill += sources[rank][task_name].prefill_us
        decode += sources[rank][task_name].decode_us
        assignments.append((task_name, rank))
        prev_rank = rank
    if prev_rank != 0:
        prefill += config.prefill_transfer_us
        decode += config.decode_transfer_us
    prefill += rank_costs[0]["norm_lm_head"].prefill_us
    decode += rank_costs[0]["norm_lm_head"].decode_us
    return PlanState(
        objective_us=prefill + config.decode_weight * decode,
        prefill_us=prefill,
        decode_us=decode,
        task_assignments=tuple(assignments),
    )


def switch_constraints(
    index: int,
    *,
    left: str | None,
    right: str | None,
) -> tuple[LinearConstraint, ...]:
    switch = switch_variable(index)
    left_terms = () if left is None else (LinearTerm(left, 1.0),)
    right_terms = () if right is None else (LinearTerm(right, 1.0),)
    return (
        LinearConstraint(
            name=f"{switch}_ge_left_minus_right",
            terms=(LinearTerm(switch, 1.0), *negate(left_terms), *right_terms),
            sense=">=",
            rhs=0.0,
        ),
        LinearConstraint(
            name=f"{switch}_ge_right_minus_left",
            terms=(LinearTerm(switch, 1.0), *left_terms, *negate(right_terms)),
            sense=">=",
            rhs=0.0,
        ),
        LinearConstraint(
            name=f"{switch}_le_left_plus_right",
            terms=(LinearTerm(switch, 1.0), *negate(left_terms), *negate(right_terms)),
            sense="<=",
            rhs=0.0,
        ),
        LinearConstraint(
            name=f"{switch}_le_two_minus_left_minus_right",
            terms=(LinearTerm(switch, 1.0), *left_terms, *right_terms),
            sense="<=",
            rhs=2.0,
        ),
    )


def fixed_task_cost(task: CsvTask, config: PlanConfig) -> float:
    return task.prefill_us + config.decode_weight * task.decode_us


def task_variable(task_name: str) -> str:
    return task_rank_variable(task_name, 1)


def task_rank_variable(task_name: str, rank: int) -> str:
    return f"x_{task_name}_rank{rank}"


def boundary_pair_variable(index: int, left_rank: int, right_rank: int) -> str:
    return f"b_boundary_{index}_r{left_rank}_r{right_rank}"


def assignment_constraint(task_name: str, *, rank_count: int) -> LinearConstraint:
    return LinearConstraint(
        name=f"assign_{task_name}",
        terms=tuple(
            LinearTerm(task_rank_variable(task_name, rank), 1.0)
            for rank in range(rank_count)
        ),
        sense="==",
        rhs=1.0,
    )


def boundary_pair_constraints(
    index: int,
    *,
    tasks: list[str],
    rank_count: int,
) -> tuple[LinearConstraint, ...]:
    constraints: list[LinearConstraint] = []
    left_task = None if index == 0 else tasks[index - 1]
    right_task = None if index == len(tasks) else tasks[index]
    for left_rank in range(rank_count):
        terms = [
            LinearTerm(boundary_pair_variable(index, left_rank, right_rank), 1.0)
            for right_rank in range(rank_count)
        ]
        rhs = 0.0
        if left_task is None:
            rhs = 1.0 if left_rank == 0 else 0.0
        else:
            terms.append(LinearTerm(task_rank_variable(left_task, left_rank), -1.0))
        constraints.append(
            LinearConstraint(
                name=f"boundary_{index}_left_rank{left_rank}",
                terms=tuple(terms),
                sense="==",
                rhs=rhs,
            )
        )
    for right_rank in range(rank_count):
        terms = [
            LinearTerm(boundary_pair_variable(index, left_rank, right_rank), 1.0)
            for left_rank in range(rank_count)
        ]
        rhs = 0.0
        if right_task is None:
            rhs = 1.0 if right_rank == 0 else 0.0
        else:
            terms.append(LinearTerm(task_rank_variable(right_task, right_rank), -1.0))
        constraints.append(
            LinearConstraint(
                name=f"boundary_{index}_right_rank{right_rank}",
                terms=tuple(terms),
                sense="==",
                rhs=rhs,
            )
        )
    return tuple(constraints)


def selected_rank_for_task(
    *,
    task_name: str,
    rank_count: int,
    values: dict[int, float],
) -> int:
    chosen = [rank for rank in range(rank_count) if values.get(rank, 0.0) > 0.5]
    if len(chosen) == 1:
        return chosen[0]
    if not values:
        raise AssertionError(f"no rank assignment values for {task_name}")
    return max(values.items(), key=lambda item: item[1])[0]


def switch_variable(index: int) -> str:
    return f"s_boundary_{index}"


def negate(terms: tuple[LinearTerm, ...]) -> tuple[LinearTerm, ...]:
    return tuple(LinearTerm(term.variable, -term.coefficient) for term in terms)


def write_milp_json(path: str | Path, model: MILPModel) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(milp_to_dict(model), indent=2) + "\n")


def write_plan_json(path: str | Path, solution: MILPSolution) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(solution_to_dict(solution), indent=2) + "\n")


def milp_to_dict(model: MILPModel) -> dict[str, Any]:
    return {
        "binary_variables": list(model.binary_variables),
        "objective": {
            "sense": model.objective.sense,
            "constant": model.objective.constant,
            "terms": [term_to_dict(term) for term in model.objective.terms],
        },
        "constraints": [
            {
                "name": constraint.name,
                "terms": [term_to_dict(term) for term in constraint.terms],
                "sense": constraint.sense,
                "rhs": constraint.rhs,
            }
            for constraint in model.constraints
        ],
        "metadata": model.metadata,
    }


def solution_to_dict(solution: MILPSolution) -> dict[str, Any]:
    plan = solution.plan
    return {
        "objective_us": plan.objective_us,
        "prefill_us": plan.prefill_us,
        "decode_us": plan.decode_us,
        "rank_switches": plan.rank_switches,
        "rank_counts": plan.rank_counts,
        "task_assignments": [
            {"name": name, "rank": rank} for name, rank in plan.task_assignments
        ],
        "milp_probe": {
            "runtime_compatible": False,
            "solver": solution.solver,
            "gurobi_status": solution.gurobi_status,
            "note": "Standalone MILP probe result via Gurobi; not connected to runtime.",
        },
    }


def term_to_dict(term: LinearTerm) -> dict[str, Any]:
    return {"variable": term.variable, "coefficient": term.coefficient}
