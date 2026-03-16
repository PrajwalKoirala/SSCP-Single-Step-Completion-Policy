# SSCP D4RL Offline RL experiments

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

## TL;DR:
This work introduces Single-Step Completion Q-Learning (SSCQL), a generative model-based policy for efficient one-shot action generation, integrating flow-matching and shortcut-completion losses for stable, effcient and expressive offline reinforcement learning.

## Install
This repo requires Python 3.11 and is based on JAX.
To install, run:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main_d4rl.py --env_name=halfcheetah-medium-v2 --offline_steps=500000 --agent.alpha2=0.20 --agent.batch_size=1024 --agent.q_agg=min --agent.normalize_q_loss=True

python main_d4rl.py --env_name=halfcheetah-medium-replay-v2 --offline_steps=500000 --agent.alpha2=0.05 --agent.batch_size=1024 --agent.q_agg=min --agent.normalize_q_loss=True

python main_d4rl.py --env_name=halfcheetah-medium-expert-v2 --offline_steps=500000 --agent.alpha2=0.1 --agent.batch_size=1024 --agent.q_agg=min --agent.normalize_q_loss=True



python main_d4rl.py --env_name=walker2d-medium-v2 --offline_steps=500000 --agent.alpha2=0.5 --agent.batch_size=1024 --agent.q_agg=min --agent.normalize_q_loss=True

python main_d4rl.py --env_name=walker2d-medium-replay-v2 --offline_steps=500000 --agent.alpha2=0.10 --agent.batch_size=1024 --agent.q_agg=min --agent.normalize_q_loss=True

python main_d4rl.py --env_name=walker2d-medium-expert-v2 --offline_steps=500000 --agent.alpha2=0.10 --agent.batch_size=1024 --agent.q_agg=min --agent.normalize_q_loss=True



python main_d4rl.py --env_name=hopper-medium-v2 --offline_steps=500000 --agent.alpha2=0.05 --agent.batch_size=1024 --agent.q_agg=min --agent.normalize_q_loss=True

python main_d4rl.py --env_name=hopper-medium-replay-v2 --offline_steps=500000 --agent.alpha2=0.10 --agent.batch_size=1024 --agent.q_agg=min --agent.normalize_q_loss=True

python main_d4rl.py --env_name=hopper-medium-expert-v2 --offline_steps=500000 --agent.alpha2=0.75 --agent.batch_size=1024 --agent.q_agg=min --agent.normalize_q_loss=True
```

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



## Acknowledgements
This work is built upon the following works:

[Flow Q-Learning](https://github.com/seohongpark/fql)

[JaxRL](https://github.com/ikostrikov/jaxrl)
