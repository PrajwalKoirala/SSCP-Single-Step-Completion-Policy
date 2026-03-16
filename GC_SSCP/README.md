# GC-SSCP OGBench Offline GCRL experiments

<div align="center">
  <a href="https://openreview.net/pdf?id=u6vDv51J9o">
    <img src="https://img.shields.io/badge/Paper-OpenReview-red" alt="Paper">
  </a>
  <a href="https://github.com/PrajwalKoirala/SSCP-Single-Step-Completion-Policy">
    <img src="https://img.shields.io/badge/Code-GitHub-green?logo=github" alt="GitHub Code">
  </a>
  <a href="https://arxiv.org/abs/2506.21427">
    <img src="https://img.shields.io/badge/Paper-Arxiv-maroon?logo=arxiv" alt="Arxiv Paper">
  </a>
  <a href="https://iclr.cc/virtual/2026/poster/10006918">
  <img src="https://img.shields.io/badge/Page-ICLR-2AA198" alt="ICLR Page">
</a>
</div>

## Installation
Install [OGBench](https://github.com/seohongpark/ogbench.git).

## Usage
Place *gc_sscp.py* and *networks.py* from GC_SSCP folder in the respective locations in OGBench for training and evaluation.

```bash
# PointMaze Navigate

python main.py --env_name=pointmaze-medium-navigate-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=25 --agent.discount=0.99

python main.py --env_name=pointmaze-large-navigate-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=100 --agent.discount=0.995

python main.py --env_name=pointmaze-giant-navigate-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=50 --agent.discount=0.995

python main.py --env_name=pointmaze-teleport-navigate-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=50 --agent.discount=0.995

```

```bash
# PointMaze Stitch

python main.py --env_name=pointmaze-medium-stitch-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=25 --agent.discount=0.99

python main.py --env_name=pointmaze-large-stitch-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=50 --agent.discount=0.995

python main.py --env_name=pointmaze-giant-stitch-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=50 --agent.discount=0.995

python main.py --env_name=pointmaze-teleport-stitch-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=50 --agent.discount=0.995

```

```bash
# AntMaze Navigate

python main.py --env_name=antmaze-medium-navigate-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=25 --agent.discount=0.99

python main.py --env_name=antmaze-large-navigate-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=100 --agent.discount=0.995

python main.py --env_name=antmaze-giant-navigate-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=100 --agent.discount=0.995

python main.py --env_name=antmaze-teleport-navigate-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=50 --agent.discount=0.995

```

```bash
# AntMaze Stitch

python main.py --env_name=antmaze-medium-stitch-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=25 --agent.discount=0.99 --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

python main.py --env_name=antmaze-large-stitch-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=50 --agent.discount=0.995 --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

python main.py --env_name=antmaze-giant-stitch-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=50 --agent.discount=0.995 --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

python main.py --env_name=antmaze-teleport-stitch-v0 --train_steps=500000 --eval_interval=50000 --agent=agents/gc_sscp.py --agent.high_alpha=5.0 --agent.low_alpha=5.0 --agent.subgoal_steps=50 --agent.discount=0.995 --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

```

<br>

## Citation
If you find this code useful, please consider citing:
```bibtex
@article{koirala2025flow,
  title={Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning},
  author={Koirala, Prajwal and Fleming, Cody},
  journal={arXiv preprint arXiv:2506.21427},
  year={2025}
}
```


