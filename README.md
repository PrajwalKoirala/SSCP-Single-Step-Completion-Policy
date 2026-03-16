# SSCP - Single Step Completion Policy

<div align="center">
  <a href="https://openreview.net/pdf?id=u6vDv51J9o" target="_blank">
    <img src="https://img.shields.io/badge/Paper-OpenReview-red" alt="Paper">
  </a>
  <a href="https://github.com/PrajwalKoirala/SSCP-Single-Step-Completion-Policy" target="_blank">
    <img src="https://img.shields.io/badge/Code-GitHub-green?logo=github" alt="GitHub Code">
  </a>
  <a href="https://arxiv.org/abs/2506.21427" target="_blank">
    <img src="https://img.shields.io/badge/Paper-Arxiv-maroon?logo=arxiv" alt="Arxiv Paper">
  </a>
  <a href="https://iclr.cc/virtual/2026/poster/10006918" target="_blank">
    <img src="https://img.shields.io/badge/Page-ICLR-2AA198" alt="ICLR Page">
  </a>
</div>


*Official Implementation of ICLR 2026 paper* ***Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning***

**Abstract:** Generative models such as diffusion and flow-matching offer expressive policies for offline reinforcement learning (RL) by capturing rich, multimodal action distributions, but their iterative sampling introduces high inference costs and training instability due to gradient propagation across sampling steps. We propose the *Single-Step Completion Policy* (SSCP), a generative policy trained with an augmented flow-matching objective to predict direct completion vectors from intermediate flow samples, enabling accurate, one-shot action generation. In an off-policy actor-critic framework, SSCP combines the expressiveness of generative models with the training and inference efficiency of unimodal policies, without requiring long backpropagation chains. Our method scales effectively to offline, offline-to-online, and online RL settings, offering substantial gains in speed and adaptability over diffusion-based baselines. We further extend SSCP to goal-conditioned RL, enabling flat policies to exploit subgoal structures without explicit hierarchical inference. SSCP achieves strong results across standard offline RL and GCRL benchmarks, positioning it as a versatile, expressive, and efficient framework for deep RL and sequential decision-making. 

**Keywords**: Offline Reinforcement Learning, Generative Models, Flow Matching, Behavior Cloning, Goal-Conditioned Reinforcement Learning

## Repository Structure

- `./SSCQL/` – [SSCQL experiments](https://github.com/PrajwalKoirala/SSCP-Single-Step-Completion-Policy/tree/main/SSCQL) with dedicated [README](https://github.com/PrajwalKoirala/SSCP-Single-Step-Completion-Policy/tree/main/SSCQL) for installation and usage.
- `./GC_SSCP/` – [GC-SSCP experiments](https://github.com/PrajwalKoirala/SSCP-Single-Step-Completion-Policy/tree/main/GC_SSCP) with dedicated [README](https://github.com/PrajwalKoirala/SSCP-Single-Step-Completion-Policy/tree/main/GC_SSCP) for installation and usage.

## Citation
If you find this repo useful, please consider [citing](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=s4CrY2wAAAAJ&citation_for_view=s4CrY2wAAAAJ:W7OEmFMy1HYC):
```bibtex
@article{koirala2025flow,
  title={Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning},
  author={Koirala, Prajwal and Fleming, Cody},
  journal={arXiv preprint arXiv:2506.21427},
  year={2025}
}
```
For inquiries, please find my up-to-date contact information [here 📫.](https://prajwalkoirala.github.io/)


