"""Small entrypoint around unchanged upstream PFLlib FedAvg."""
import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'third_party/PFLlib/system'))
sys.path.insert(0, str(ROOT))
from flcore.servers.serveravg import FedAvg
from flcore.trainmodel.models import FedAvgCNN
from src.eval.common import state_hash


class RecordedFedAvg(FedAvg):
    def __init__(self, *args):
        self.selections = []
        super().__init__(*args)

    def select_clients(self):
        selected = super().select_clients()
        self.selections.append([c.id for c in selected])
        return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    cfg = json.loads((ROOT/'configs/pfllib_100c.json').read_text())
    cfg.update(dataset=a.dataset, commit=a.commit, smoke=a.smoke, synthetic=a.synthetic)
    rounds = 2 if a.smoke or a.synthetic else cfg['rounds']
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(a.data_root).resolve()
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available(), 'GPU run requested'
    if a.synthetic:
        for split in ('train','test'):
            folder = data_root/a.dataset/split
            folder.mkdir(parents=True,exist_ok=True)
            for cid in range(100):
                np.savez_compressed(folder/f'{cid}.npz',data={'x':np.random.randn(20,1,28,28).astype('float32'),'y':np.arange(20)%10})
    (out/'system').mkdir(exist_ok=True)
    (out/'dataset').symlink_to(data_root, target_is_directory=True)
    os.chdir(out/'system')
    classes = {'MNIST':10,'Cifar10':10,'Cifar100':100,'TinyImagenet':200,'PFLsmoke':10}[a.dataset]
    channels, dim = (1,1024) if a.dataset in ('MNIST','PFLsmoke') else (3,10816 if a.dataset=='TinyImagenet' else 1600)
    model = FedAvgCNN(channels, classes, dim).cuda()
    args = SimpleNamespace(device='cuda',dataset=a.dataset,num_classes=classes,
        global_rounds=rounds-1,local_epochs=cfg['local_epochs'],batch_size=cfg['batch_size'],
        local_learning_rate=cfg['local_learning_rate'],model=model,num_clients=100,
        join_ratio=.1,random_join_ratio=False,few_shot=0,algorithm='FedAvg',
        time_select=False,goal='TTFL',time_threthold=1e12,save_folder_name=str(out/'models'),
        top_cnt=100,auto_break=False,eval_gap=cfg['eval_gap'],client_drop_rate=0.,
        train_slow_rate=0.,send_slow_rate=0.,dlg_eval=False,dlg_gap=100,
        batch_num_per_client=1,num_new_clients=0,fine_tuning_epoch_new=0,
        learning_rate_decay=False,learning_rate_decay_gamma=.99)
    cfg.update(actual_rounds=rounds,global_rounds_argument=rounds-1,num_classes=classes,
               in_features=channels,dim=dim,upstream_revision='0169ba7e412c9856a08bb3faefab1e35f538a3c1')
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    server = RecordedFedAvg(args,0)
    initial = state_hash(server.global_model)
    t = time.time()
    server.train()
    # Upstream evaluation is pre-update. Broadcast final aggregate for final metric.
    server.send_models()
    ids,counts,correct,auc = server.test_metrics()
    torch.save(server.global_model.state_dict(),out/'global_state.pt')
    final = state_hash(server.global_model)
    assert len(server.selections) == rounds
    assert all(len(set(s)) == 10 for s in server.selections)
    assert initial != final
    result = {'final_accuracy':100*sum(correct)/sum(counts), 'client_accuracy':[100*c/n for c,n in zip(correct,counts)],
              'client_ids':ids,'client_test_samples':counts,'test_correct':correct,
              'elapsed_seconds':time.time()-t,'selection_history':server.selections,
              'initial_hash':initial,'final_hash':final,'model_device':str(next(server.global_model.parameters()).device),
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'python':platform.python_version(),
              'baseline_history':server.rs_test_acc,'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
              'rounds':rounds,'clients':100,'participants_each_round':10}
    (out/'baseline.json').write_text(json.dumps(result,indent=2))
    print('PFL_DONE',a.dataset,result['final_accuracy'],rounds,flush=True)


if __name__ == '__main__':
    main()
