# PDF / Paper Manifest

This file tracks source papers and where to fetch them. Binary PDFs are not committed by default; fetch them into `runs/papers/` or another scratch path when deep reading requires local annotation.

| Work | PDF / HTML | Local status |
| --- | --- | --- |
| NMR | https://arxiv.org/pdf/2603.22201 | Fetched to `runs/papers/nmr-2603.22201.pdf` |
| PDF-HR | https://arxiv.org/pdf/2602.04851 | Fetched to `runs/papers/pdfhr-2602.04851.pdf` |
| Retargeting Matters / GMR | https://arxiv.org/pdf/2510.02252 | Fetched to `runs/papers/gmr-2510.02252.pdf` |
| BeyondMimic | https://arxiv.org/pdf/2508.08241 | Link recorded |
| OmniTrack | https://arxiv.org/pdf/2602.23832 | Fetched to `runs/papers/omnitrack-2602.23832.pdf` |
| ULTRA | https://arxiv.org/pdf/2603.03279 | Link recorded |
| OmniRetarget | https://arxiv.org/pdf/2509.26633 | Fetched to `runs/papers/omniretarget-2509.26633.pdf` |
| ReActor | https://arxiv.org/pdf/2605.06593 | Fetched to `runs/papers/reactor-2605.06593.pdf` |
| PHUMA | https://arxiv.org/pdf/2510.26236 | Fetched to `runs/papers/phuma-2510.26236.pdf` |
| ExBody2 | https://arxiv.org/pdf/2412.13196 | Link recorded |
| KungfuBot | https://arxiv.org/pdf/2506.12851 | Fetched to `runs/papers/kungfubot-2506.12851.pdf`; OpenAlex resolved this title to arXiv `2506.12851` |
| RoboForge | https://arxiv.org/pdf/2603.17927 | Fetched to `runs/papers/roboforge-2603.17927.pdf` |
| DynaRetarget | https://arxiv.org/pdf/2602.06827 | Fetched to `runs/papers/dynaretarget-2602.06827.pdf`; OpenAlex resolved as `W7128373084` / `W7128408694` |
| SPIDER | https://arxiv.org/pdf/2511.09484 | Fetched to `runs/papers/spider-2511.09484.pdf`; OpenAlex resolved as `W4416307812` |
| KDMR | https://arxiv.org/pdf/2603.09956 | Fetched to `runs/papers/kdmr-2603.09956.pdf` |
| Contact and Dynamics from Monocular Video | https://davrempe.github.io/docs/contact-and-dynamics-2020.pdf | Link recorded |
| Contact-aware Motion Retargeting / self-contact retargeting | https://arxiv.org/pdf/2109.07431 | Link recorded |
| UNDERPRESSURE foot contact dataset | https://inria.hal.science/hal-03865772/file/SCA_2022_UnderPressure.pdf | Link recorded |
| Shared Latent Retargeting | https://www.roboticsproceedings.org/rss16/p071.pdf | Link recorded |

Fetch convention:

```bash
mkdir -p runs/papers
curl -L https://arxiv.org/pdf/2603.22201 -o runs/papers/nmr-2603.22201.pdf
```
