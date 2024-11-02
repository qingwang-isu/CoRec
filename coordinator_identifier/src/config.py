from transformers import BertTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 5
BASE_MODEL_PATH = "../bert-base-uncased"
MODEL_PATH = "saved_models/model_c.bin"
TRAINING_FILE = "../data/wsj/wsj_train_c.csv"
DEVELOPMENT_FILE = "../data/wsj/wsj_dev_c.csv"
TEST_FILE = "../data/wsj//wsj_test_c.csv"
TOKENIZER = BertTokenizer.from_pretrained(BASE_MODEL_PATH, do_lower_case=True)
