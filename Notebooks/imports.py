import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pdfplumber
import nltk
import spacy
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm
import sqlite3
import re
from pathlib import Path
from zipfile import ZipFile


nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")

nlp_2 = English()
nlp_2.add_pipe("sentencizer")

DATAFILES_PATH = "../Data"