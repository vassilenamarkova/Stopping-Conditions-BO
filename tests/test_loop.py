import torch
from bo.objective import sine, SINE_BOUNDS
from bo.loop import run_bo_loop


def test_loop_runs_without_error():
    train_x, train_y, best_so_far = run_bo_loop(
        sine, SINE_BOUNDS, n_init=4, n_iters=3, seed=1,
    )
    assert train_x.shape[0] == 4 + 3  
    assert train_y.shape[0] == 4 + 3
    assert len(best_so_far) == 3 + 1  

def test_best_so_far_is_non_decreasing():
    _, _, best_so_far = run_bo_loop(
        sine, SINE_BOUNDS, n_init=4, n_iters=5, seed=1,
    )
    for i in range(1, len(best_so_far)):
        assert best_so_far[i] >= best_so_far[i - 1]


def test_loop_is_reproducible_with_same_seed():
    _, _, best_1 = run_bo_loop(sine, SINE_BOUNDS, n_init=4, n_iters=3, seed=42)
    _, _, best_2 = run_bo_loop(sine, SINE_BOUNDS, n_init=4, n_iters=3, seed=42)
    assert best_1 == best_2