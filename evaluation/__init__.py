"""
ChatGIT Evaluation Package.

Modules:
  eval_retrieval      — MRR, Recall, NDCG, P@k, MAP, Success@k
  eval_generation     — CodeBLEU, ROUGE-L, BERTScore, Exact Match, Pass@1
  eval_conversation   — Multi-turn CCS, coreference accuracy, redundancy rate
  eval_faithfulness   — RAGAS-style faithfulness, context precision/recall
  statistical_tests   — Wilcoxon, bootstrap CI, Cohen's d, Fleiss' κ
  baselines           — BM25, VanillaRAG, ConvAwareRAG, BM25-SlidingWindow, GraphRAG-Code
  ablation            — Per-novelty ablation framework
  human_eval_protocol — Human annotation schema and IAA
  run_benchmark       — Master pipeline runner
"""
