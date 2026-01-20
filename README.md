# Contrasting Linguistic Patterns in Human and LLM-Generated News Text

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/Springer-Paper-blue.svg)](https://link.springer.com/article/10.1007/s10462-024-10903-2)

This repository contains the official implementation for the paper:
**[Contrasting Linguistic Patterns in Human and LLM-Generated News Text](https://link.springer.com/article/10.1007/s10462-024-10903-2)**
*Alberto Muñoz-Ortiz, Carlos Gómez-Rodríguez, and David Vilares*
Published in **Artificial Intelligence Review**, Volume 57, Article 265 (2024).

---

## Overview

This project conducts a quantitative analysis contrasting human-written English news text with comparable large language model (LLM) output from six different LLMs covering three families and four sizes. The analysis spans several measurable linguistic dimensions, including morphological, syntactic, psychometric, and sociolinguistic aspects.

Key findings include:
- Human texts exhibit more scattered sentence length distributions and vocabulary variety.
- Humans show distinct use of dependency and constituent types, with shorter constituents and more optimized dependency distances.
- Humans tend to exhibit stronger negative emotions compared to LLMs.
- LLM outputs use more numbers, symbols, and auxiliaries than human texts.
- Sexist bias prevalent in human text is also expressed by LLMs, and even magnified in most of them.

## Project Structure

The repository is organized as follows:

- `src/`: Core analysis utilities.
  - `utils.py`: CoNLLu parsing and dependency statistics.
  - `constituency.py`: Constituency tree analysis.
  - `analysis_f.py`: Additional analysis functions.
- `scripts/`: Entry-point scripts.
  - `download_articles.py`: Download NYT articles via API.
  - `generate.py`: Generate text using various LLMs.
  - `parse_classify.py`: Parse and classify articles.
- `notebooks/`: Jupyter notebooks for analysis and visualization.
  - `analysis.ipynb`: Main analysis notebook.
- `data/`: Generated articles from different LLMs.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amunozo/linguistic_patterns_LLMs.git
   cd linguistic_patterns_LLMs
   ```

2. **Install dependencies:**
   ```bash
   pip install transformers stanza torch pandas tqdm requests
   ```

3. **Setup NYT API Key** (for downloading articles):
   - Get your API key from [NYT Developer Portal](https://developer.nytimes.com/get-started)
   - Copy `.env.example` to `.env` and add your key
   - **Important**: Never commit your `.env` file

## Usage

### Downloading Articles
```bash
python scripts/download_articles.py 2023-10 2024-01 data/nyt/
```

### Generating LLM Text
```bash
python scripts/generate.py
```

### Parsing and Classification
```bash
python scripts/parse_classify.py
```

## Citation

```bibtex
@article{munoz-ortiz-etal-2024-contrasting,
    title = "Contrasting Linguistic Patterns in Human and LLM-Generated News Text",
    author = "Mu{\~n}oz-Ortiz, Alberto and
      G{\\'o}mez-Rodr{\\'i}guez, Carlos and
      Vilares, David",
    journal = "Artificial Intelligence Review",
    volume = "57",
    number = "10",
    pages = "265",
    year = "2024",
    publisher = "Springer Science and Business Media LLC",
    doi = "10.1007/s10462-024-10903-2",
    url = "https://link.springer.com/article/10.1007/s10462-024-10903-2",
}
```

## Contact

**Alberto Muñoz-Ortiz** - [alberto.munoz.ortiz@udc.es](mailto:alberto.munoz.ortiz@udc.es)

## Acknowledgments

Open Access funding provided thanks to the CRUE-CSIC agreement with Springer Nature. We acknowledge the European Research Council (ERC), which has funded this research under the Horizon Europe research and innovation programme (SALSA, grant agreement No 101100615); SCANNER-UDC (PID2020-113230RB-C21) funded by MICIU/AEI/10.13039/501100011033; Xunta de Galicia (ED431C 2020/11); GAP (PID2022-139308OA-I00) funded by MICIU/AEI/10.13039/501100011033/ and by ERDF, EU; Grant PRE2021-097001 funded by MICIU/AEI/10.13039/501100011033 and by ESF+ (predoctoral training grant associated to project PID2020–113230RB-C21); and Centro de Investigación de Galicia "CITIC", funded by the Xunta de Galicia through the collaboration agreement between the Consellería de Cultura, Educación, Formación Profesional e Universidades and the Galician universities for the reinforcement of the research centres of the Galician University System (CIGUS). Funding for open access charge: Universidade da Coruña/CISUG.
