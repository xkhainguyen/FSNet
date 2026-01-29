# Synthetic Constrained Optimization

## 🚀 Installation

Install dependencies:
```bash
conda env create -f ml4opt.yml
```

## 🎓 Usage

### Training and Test

* `--method`

  * `FSNet`              (Feasibility-Seeking Neural Network)
  * `penalty`            (Penalty method)
  * `adaptive_penalty`   (Adaptive Penalty method)
  * `DC3`                (Deep Constraint Completion and Correction)

**Baseline:**
```bash
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
```

**Our framework:**

Cheap labels with different quality levels are provided in `/datasets`

Then we pretrain a supervised warm-start
```bash
    python main.py \
        --method sup_pen \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --train_size 7000 \
        --en_subopt 3 \
        --subopt_ratio 0.5 \
        --save_intermediate True
```

Use `pick_ckpt_merit.ipynb` to pick the best checkpoint for the next step.

Then we train with normal SSL with your checkpoint
```bash
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --checkpoint [your_check_point]
```