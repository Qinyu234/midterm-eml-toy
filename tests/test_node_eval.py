from codec.node import is_evaluable
from eval.node_eval import NodeEvaluator, generate_report
from data.feynman import get_feynman_function, list_feynman_functions


def test_node_evaluator_init():
    ev = NodeEvaluator(steps=5)
    assert ev.steps == 5


def test_skip_empty_node():
    ev = NodeEvaluator(steps=3, functions=[get_feynman_function('I.6.2')])
    r = ev.evaluate_single_node(0, get_feynman_function('I.6.2'))
    assert not r.converged
    assert r.final_rmse == float('inf')


def test_evaluate_single_node():
    ev = NodeEvaluator(steps=5, functions=[get_feynman_function('I.8.4')])
    r = ev.evaluate_single_node(1, get_feynman_function('I.8.4'))
    assert r.byte == 1


def test_node_evaluation():
    ev = NodeEvaluator(steps=3, functions=list_feynman_functions(n_vars=1)[:2])
    results = ev.evaluate(bytes=[1, 2])
    assert len(results) == 4


def test_adaptability_matrix():
    ev = NodeEvaluator(steps=3, functions=[get_feynman_function('I.6.2')])
    m = ev.adaptability_matrix(bytes=[1, 2])
    assert m.rmse.shape == (2, 1)


def test_generate_report():
    ev = NodeEvaluator(steps=3, functions=[get_feynman_function('I.6.2')])
    m = ev.adaptability_matrix(bytes=[1])
    report = generate_report(m)
    assert 'Adaptability' in report
