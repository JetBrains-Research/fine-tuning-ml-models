from .preprocess import run_psiminer
from scripts.utils import PSIMINER_CODE2SEQ_TOPIC_CONFIG

run_psiminer("java-med/training/airbnb__airpal", "dataset", PSIMINER_CODE2SEQ_TOPIC_CONFIG)
