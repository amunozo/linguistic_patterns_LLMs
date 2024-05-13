# Contrasting Linguistic Patterns in Humans and LLMs-Generated News Text
The code in this repository corresponds to the paper "Contrasting Linguistic Patterns in Humans and LLMs-Generated News Text, under review at the journal "Artificial Intelligence Review".

### Reproduction of results
As obtaining the NYT articles need an API key, we decided not to upload the news used on the article directly to a repository. However, they are easily recoverable using the script "download_articles.py".
The starting date is October 1, 2023, and the final date is January 24, 2024. The user should discard the articles from the rest of the days of January.

The file "generates_articles.py" can be used to generate the texts. Downloading LLaMa locally and building the model is required, as the weights are not freely available on HuggingFace.
Our generated articles are stored in the data folder.
