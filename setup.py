from setuptools import setup
import subprocess

subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])

setup(
    name='project',
    version='0.1',
    install_requires=[
        'spacy',
    ],
)
