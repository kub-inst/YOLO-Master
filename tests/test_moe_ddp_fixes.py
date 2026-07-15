"""Regression contracts for trainer sparse-MoE DDP safety."""
import ast
from pathlib import Path
S=(Path(__file__).parents[1]/'ultralytics/engine/trainer.py').read_text()
def test_syntax(): ast.parse(S)
def test_contracts():
 assert 'torch.cuda.set_device(LOCAL_RANK)' in S; assert 'backend="nccl"' in S
 assert 'device_ids=[LOCAL_RANK]' in S; assert 'find_unused_parameters=True' in S
 assert 'broadcast_buffers=False' in S and 'static_graph=True' in S
 assert 'amp_flag.item()' in S; assert 'self.batch_size % self.world_size' in S
def test_accumulation_and_collapse():
 assert 'or i == nb - 1' in S; assert 'self.model.no_sync()' in S; assert 'if should_step:' in S
