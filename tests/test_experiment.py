from dataclasses import dataclass, field
from typing import List

from experiment.metrics import SearchRun, StepRecord, recovery_rate
from experiment.search import search_run
from data.feynman import get_feynman_function


@dataclass
class ExperimentGroup:
    name: str
    runs: List[SearchRun] = field(default_factory=list)


def test_step_record():
    r = StepRecord(step=1, loss=0.5)
    assert r.step == 1


def test_mean_ae():
    run = SearchRun(byte=1, eq_id='I.6.2')
    run.add_record(0, 1.0)
    run.add_record(1, 0.5)
    run.add_record(2, 0.25)
    assert run.mean_ae > 0


def test_recovery_rate():
    runs = [
        SearchRun(1, 'a', recovered=True),
        SearchRun(2, 'a', recovered=False),
    ]
    assert recovery_rate(runs) == 0.5


def test_search_run():
    fn = get_feynman_function('I.8.4')
    run = search_run(1, fn, steps=10)
    assert len(run.records) == 10
    assert run.eq_id == 'I.8.4'


def test_search_run_ae():
    fn = get_feynman_function('I.6.2')
    run = search_run(1, fn, steps=5)
    assert isinstance(run.mean_ae, float)


def test_experiment_group():
    g = ExperimentGroup('test')
    g.runs.append(SearchRun(1, 'I.6.2'))
    assert len(g.runs) == 1
