# PDF / Paper Manifest

This file tracks source papers and where to fetch them. Binary PDFs are not committed by default; fetch them into `runs/papers/` or another scratch path when deep reading requires local annotation.

| Work | PDF / HTML | Local status |
| --- | --- | --- |
| NMR | https://arxiv.org/pdf/2603.22201 | Link recorded |
| PDF-HR | https://arxiv.org/pdf/2602.04851 | Link recorded |
| Retargeting Matters / GMR | https://arxiv.org/pdf/2510.02252 | Link recorded |
| BeyondMimic | https://arxiv.org/pdf/2508.08241 | Link recorded |
| OmniTrack | https://arxiv.org/pdf/2602.23832 | Link recorded |
| ULTRA | https://arxiv.org/pdf/2603.03279 | Link recorded |
| OmniRetarget | https://arxiv.org/pdf/2509.26633 | Link recorded |
| ReActor | https://arxiv.org/pdf/2605.06593 | Link recorded |
| PHUMA | https://arxiv.org/pdf/2510.26236 | Link recorded |
| ExBody2 | https://arxiv.org/pdf/2412.13196 | Link recorded |
| KungFuAthlete | https://arxiv.org/pdf/2602.13656 | Link recorded |
| KDMR | https://arxiv.org/pdf/2603.09956 | Link recorded |
| Contact and Dynamics from Monocular Video | https://davrempe.github.io/docs/contact-and-dynamics-2020.pdf | Link recorded |
| Contact-aware Motion Retargeting / self-contact retargeting | https://arxiv.org/pdf/2109.07431 | Link recorded |
| UNDERPRESSURE foot contact dataset | https://inria.hal.science/hal-03865772/file/SCA_2022_UnderPressure.pdf | Link recorded |
| Shared Latent Retargeting | https://www.roboticsproceedings.org/rss16/p071.pdf | Link recorded |

Fetch convention:

```bash
mkdir -p runs/papers
curl -L https://arxiv.org/pdf/2603.22201 -o runs/papers/nmr-2603.22201.pdf
```
