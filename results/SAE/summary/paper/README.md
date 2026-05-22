# SAE findings — ACM sigconf paper (2 pages)

`main.tex` (acmart, `sigconf`) + `references.bib` + `images/`. The compiled
`main.pdf` is the 2-page findings paper (also copied to
`../SAE_findings_paper.pdf`). All numbers come from `results/SAE/json/` metrics.

Compile on the cluster:
```bash
module load texlive/2026
cd results/SAE/summary/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Or upload the folder to Overleaf (acmart is built in). Figures in `images/`:
`fig_summary.png` (6-panel composite), `fig_depth.png`, `fig_sufficiency.png`.
