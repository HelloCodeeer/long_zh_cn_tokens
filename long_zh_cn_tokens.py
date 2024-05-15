import tiktoken
import langdetect

from tqdm import tqdm
from loguru import logger


model = "gpt-4o"
encoding_name = tiktoken.encoding_name_for_model(model)
logger.info(f"Model: {model}, Encoding: {encoding_name}")
logger.add(f"{model}_{encoding_name}.log", format="{time} {level} {message}", level="INFO", rotation="10 MB")


T = tiktoken.encoding_for_model(model)
length_dict = {}


logger.info(f"Total vocab: {T.n_vocab}")
for i in tqdm(range(T.n_vocab)):
    try:
        length_dict[i] = len(T.decode([i]))
    except:
        pass


# Sort by length
length_dict = dict(sorted(length_dict.items(), key=lambda item: -item[1]))


# Print the top 100 chinese words
tot = 0
total_len = T.n_vocab
logger.info("Top 100 Chinese words:")
for idx, item in enumerate(tqdm(length_dict)):
    try:
        if langdetect.detect(T.decode([item])) == "zh-cn":
            tot += 1
            data = {item: T.decode([item])}
            logger.info(f"{tot}({idx+1}/{total_len}): {data}")
    except:
        pass
    if tot == 100:
        break