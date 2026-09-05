"""Active-learning self-healing loop for neural adapter re-distillation."""

from typing import Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.schema.loader import get_default_backend
from paw_kit.test.runner import TestRunReport, TestRunner
from paw_kit.test.suite import TestSuiteConfig


class ActiveLearningReport(BaseModel):
    """Execution summary of the active-learning auto-repair cycle."""

    task_name: str
    adapter_path: str
    iterations_run: int = 0
    final_pass_rate: float = 0.0
    recompiled: bool = False
    repaired_edge_cases: int = 0
    is_success: bool = False
    iteration_reports: List[TestRunReport] = Field(default_factory=list)


def run_active_learning_loop(
    config: TestSuiteConfig,
    backend: Optional[AbstractPAWBackend] = None,
    teacher_provider: Optional[Callable[[str], str]] = None,
    initial_dataset: Optional[List[Dict[str, str]]] = None,
) -> ActiveLearningReport:
    """Execute the active-learning self-healing loop on a .paw adapter.

    Tests the adapter against adversarial fuzzing and assertions. If failures occur,
    queries teacher_provider for authoritative ground truth, augments the training set,
    recompiles the adapter, and re-tests until 100% compliance or max_iterations reached.

    Args:
        config: Loaded TestSuiteConfig.
        backend: PAW backend implementation for inference and recompilation.
        teacher_provider: Callable querying frontier teacher model for gold labels.
        initial_dataset: Optional baseline training dataset to augment.

    Returns:
        ActiveLearningReport with iteration metrics and final compliance status.
    """
    active_backend = backend or get_default_backend()
    runner = TestRunner(backend=active_backend)
    dataset: List[Dict[str, str]] = list(initial_dataset or [])

    # Seed dataset with standard cases if not already provided
    if not dataset:
        for sc in config.standard_cases:
            if sc.expected is not None:
                dataset.append({"input": sc.input, "output": sc.expected})

    max_iter = config.active_learning.max_iterations if config.active_learning.auto_recompile else 1
    iteration_reports: List[TestRunReport] = []
    recompiled = False
    total_repaired = 0

    for iteration in range(1, max_iter + 1):
        # 1. Run test suite
        report = runner.run(config)
        iteration_reports.append(report)

        # 2. Check if 100% assertions pass
        if report.is_success:
            return ActiveLearningReport(
                task_name=config.task_name,
                adapter_path=config.adapter_path,
                iterations_run=iteration,
                final_pass_rate=report.pass_rate,
                recompiled=recompiled,
                repaired_edge_cases=total_repaired,
                is_success=True,
                iteration_reports=iteration_reports,
            )

        # 3. Handle failures: If auto_recompile enabled and iterations remain, repair
        if not config.active_learning.auto_recompile or iteration >= max_iter:
            break

        if teacher_provider is None:
            raise ValueError(
                "Active learning auto-recompilation requires a valid teacher_provider callable."
            )

        failing = report.get_failing_inputs()
        newly_repaired = 0
        for inp, _out, _reasons in failing:
            # Query teacher for gold label
            gold_label = teacher_provider(inp)
            dataset.append({"input": inp, "output": gold_label})
            newly_repaired += 1

        total_repaired += newly_repaired

        # 4. Trigger re-compilation with augmented dataset
        active_backend.compile(
            spec=config.spec,
            examples=dataset,
            output_path=config.adapter_path,
        )
        recompiled = True

    final_report = iteration_reports[-1]
    return ActiveLearningReport(
        task_name=config.task_name,
        adapter_path=config.adapter_path,
        iterations_run=len(iteration_reports),
        final_pass_rate=final_report.pass_rate,
        recompiled=recompiled,
        repaired_edge_cases=total_repaired,
        is_success=final_report.is_success,
        iteration_reports=iteration_reports,
    )
