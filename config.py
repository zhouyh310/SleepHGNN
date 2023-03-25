from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any


@dataclass(frozen=True)
class ConstFields:
    n_subjects: int = 10
    n_node_types: int = 4  # [EEG, EOG, EMG, ECG]
    n_relation_types: int = 16

    
@dataclass
class BaseConfig:
    const: ConstFields = field(default_factory=ConstFields)

    task_name: str = MISSING

    hydra: Any = field(default_factory=lambda: {
        'run': {
            'dir': r'outputs/${task_name}'
        }
    })

    data_root: str = './data'
    feature_dirname: str = 'psd'
    label_dirname: str = 'label'
    adj_mat_dirname: str = 'nmi_adj_mat/threshold_0.1'

    output_root: str = r'outputs/${task_name}/.task'
    criterion_root: str = r'${output_root}/criterion'
    plot_root: str = r'${output_root}/plot'

    shuffle: bool = True
    k_fold: int = 10
    max_epochs: int = 200
    batch_size: int = 2048
    lr: float = 5e-4
    l2_decay: float = 1e-3
    n_HGTs: int = 3
    n_heads: int = 8
    emb_dim: int = 128
    lin_dims: List[int] = field(default_factory=lambda: [512, 128])
    lin_dropout: float = 0.2


@dataclass
class MyConfig(BaseConfig):
    '''
        Name the task and override the other settings (here, or in command line) that you want to modify.
    '''
    task_name: str = MISSING