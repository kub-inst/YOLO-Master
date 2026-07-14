from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest, torch
from torch import nn
from ultralytics.engine.trainer import BaseTrainer
class E(nn.Module):
 def __init__(self,p=False): super().__init__(); self.register_buffer('diagnostic',torch.tensor(1.),persistent=p)
def tr(m): t=object.__new__(BaseTrainer);t.ema=SimpleNamespace(ema=m);t.world_size=2;return t
def test_nccl_skips_nonpersistent_cpu():
 t=tr(E(False))
 with patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_backend',return_value='nccl'),patch('torch.distributed.broadcast') as b:t._sync_ema_buffers_for_validation()
 b.assert_not_called()
def test_nccl_rejects_persistent_cpu():
 t=tr(E(True))
 with patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_backend',return_value='nccl'),pytest.raises(RuntimeError,match='Persistent EMA buffer'):t._sync_ema_buffers_for_validation()
def test_train_destroys_group_on_error():
 t=object.__new__(BaseTrainer);t.ddp=False;t._do_train=MagicMock(side_effect=RuntimeError('boom'))
 with patch('torch.distributed.is_available',return_value=True),patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.destroy_process_group') as d,pytest.raises(RuntimeError):t.train()
 d.assert_called_once_with()
def test_nonfinite_without_checkpoint_fails():
 t=object.__new__(BaseTrainer);t.loss=torch.tensor(float('nan'));t.fitness=1.;t.best_fitness=1.;t.start_epoch=0;t.last=MagicMock();t.last.exists.return_value=False;t.device=torch.device('cpu')
 with pytest.raises(RuntimeError,match='without a healthy recovery checkpoint'):t._handle_nan_recovery(0)
