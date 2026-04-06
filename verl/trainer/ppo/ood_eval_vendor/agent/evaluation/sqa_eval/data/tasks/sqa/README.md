Place the local SQA rubric files in this directory before running offline SQA evaluation.

Required files:
- `rubrics_v1_recomputed.json`
- `rubrics_v2_recomputed.json`

Current code will look for them here by default:
- `verl/verl/trainer/ppo/ood_eval_vendor/agent/evaluation/sqa_eval/data/tasks/sqa/`

Do not create empty placeholder JSON files. The evaluator will try to load them as real rubric data.
