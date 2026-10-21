# LipScribe: Technical Implementation and System Development Document

LaTeX source for the implementation document of **LipScribe: Lip-Assisted Speech
Recognition for Robust Live Captioning** (Computer Vision and Deep Learning,
Group 4, Uganda Christian University).

## Contents

```
implementation_document/
├── main.tex                 # preamble, document structure, \input of sections
├── Implementation plan 1.pdf  # compiled document (current version)
├── Implementation plan.pdf    # earlier compiled version
├── README.md
└── sections/
    ├── titlepage.tex
    ├── abbreviations.tex     # list of abbreviations + statement labels
    ├── 01_introduction.tex   ... 15_conclusion.tex   # one file per chapter
    └── references.tex        # bibliography (thebibliography, no BibTeX step)
```

All diagrams (architecture, workflow, live-captioning threads, Gantt chart) are
drawn with TikZ inside the section files, so there are no external figure files.
The bibliography is written directly with `thebibliography`, so no
`references.bib` or BibTeX/Biber run is needed. References [1]–[8] reproduce the
proposal's bibliography; [9]–[18] were added for this document and are listed
separately.

## Compiling

Any standard TeX distribution (TeX Live 2022+ or MiKTeX) works. Run the
commands from this directory.

**pdfLaTeX** (run twice so the table of contents and cross-references resolve):

```sh
pdflatex main.tex
pdflatex main.tex
```

**latexmk** (reruns automatically):

```sh
latexmk -pdf main.tex
```

XeLaTeX and LuaLaTeX also work (`latexmk -xelatex main.tex`).

**Tectonic** (used to build the supplied PDF; it downloads the
packages it needs on first use and reruns automatically):

```sh
tectonic main.tex
```

### Required packages

`iftex, lmodern, microtype, geometry, setspace, parskip, titlesec, fancyhdr,
enumitem, amsmath, amssymb, bm, graphicx, booktabs, tabularx, array, longtable,
multirow, float, caption, xcolor, tikz, listings, algorithm, algpseudocode
(algorithmicx), hyperref, cleveref`. All of them ship with full TeX Live and
MiKTeX installations; MiKTeX installs missing ones on demand.

## Build status

The supplied `Implementation plan 1.pdf` (43 pages: 5 front-matter + 38 numbered) was built with
Tectonic 0.15.0 (XeTeX engine) on Windows. The final build reported no errors,
no undefined references or citations, and no overfull boxes. Some underfull-box
warnings remain in narrow table columns; they do not affect the output. The
build has not been tested with pdfLaTeX, but the preamble selects the right
font encoding for each engine.

Compiling writes `main.pdf` next to `main.tex`. Build outputs (`main.pdf`,
`*.aux`, `*.log`, `*.toc` and so on) are ignored by Git; rename the PDF if you
want to commit a new version.

## Editing notes

- Statement tags: `\spec` = specified by the proposal, `\rec` = recommended
  implementation choice, `\tbd` = to be decided. Keep using them when you add
  content so specifications and recommendations stay distinguishable.
- The document reports **no results**. Table 12 (`tab:results-template`) is an
  empty template to fill in once experiments have run.
