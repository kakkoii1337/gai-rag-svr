which python
source ~/.venv/bin/activate
pip install -e .
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab
python -m nltk.downloader averaged_perceptron_tagger_eng
echo "gai-sdk version:"
echo pip list | grep gai-sdk