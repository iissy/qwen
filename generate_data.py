import pyarrow as pa
import pyarrow.parquet as pq
import ujson
import re
from bs4 import BeautifulSoup


def split_txt_cropus_to_chunk_data(
    texts: list, batch_size: int = 512**2, max_len: int = 512, window_size: int = 2
) -> list:

    buffer, buffer_len = [], 0
    chunk_data = []

    for i, line in enumerate(texts):
        buffer_len += len(line)
        buffer.append(line)

        if buffer_len >= batch_size or i == len(texts) - 1:
            buffer_txt = "".join(buffer)

            # - window_size为滑动窗口，这样每个窗口都包含有window_size个上文
            for i in range(0, len(buffer_txt), max_len - window_size):

                chunk_data.append("".join(buffer_txt[i : i + max_len]))
                
                print("{}------------------",i)
                print(chunk_data)

            buffer, buffer_len = [], 0

    return chunk_data


def process_none(s: str) -> str:
    if s:
        return s
    return ""


def gen_wiki_filter(origin_file, output_file="../datasets/wiki_fi.parquet"):
    lines = []
    with open(origin_file, "r", encoding="utf-8") as f:
        items = ujson.load(f)
        i = 0
        for item in items:
            lines.append(item["completion"] + "<|im_end|>")
            i += 1
            if i > 10:
                break
    chunk_data = split_txt_cropus_to_chunk_data(lines)
    tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
    pq.write_table(
        table=tb,
        where=output_file,
        row_group_size=50000,
        data_page_size=50000,
    )


def gen_aplca_sft(origin_file, output_file):
    lines = []
    with open(origin_file, "r", encoding="utf-8") as f:
        items = ujson.load(f)

        for item in items:
            if "output" not in item.keys():
                continue
            txt = f"{item['instruction']}{item['output']}"
            if len(txt) == 0 or len(txt) > 512:
                continue
            lines.append(item)
    # print(lines[0])
    tb = pa.Table.from_pylist(lines)
    pq.write_table(
        table=tb,
        where=output_file,
        row_group_size=20480,
        data_page_size=20480,
    )


def gen_wiki(origin_file, output_file):
    lines = []
    with open(origin_file, "r", encoding="utf-8") as f:
        for line in f:
            item = ujson.loads(line)
            content = item["text"].replace('\n', '')
            content = remove_all_tags(content)
            if len(content) == 0:
                print(item["id"])
                continue
            content = content + "<|im_end|>"
            if 100 <= len(content) <= 512:
                lines.append(content)
                # print(item["id"])

    print(len(lines))
    tb = pa.Table.from_arrays(arrays=[pa.array(lines)], names=["text"])
    pq.write_table(
        table=tb,
        where=output_file,
        row_group_size=50000,
        data_page_size=50000,
    )


def remove_hr_tags(html_content):
    hr_pattern = re.compile(r'<hr[^>]*>', re.IGNORECASE)
    cleaned_content = re.sub(hr_pattern, '', html_content)
    return cleaned_content


def remove_all_tags(html_content):
    if html_content.find("<") == -1:
        return html_content

    try:
        soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
        for tag in soup.find_all():
            tag.decompose()

        return str(soup)
    except Exception as e:
        print(html_content)
        print(str(e))
        return ""


def gen_alpaca_train_sft(output_file):
    lines = []
    for file in [
        "../data/alpaca_data_zh_51k.json",
        "../data/sft_train.json",
    ]:
        with open(file, "r", encoding="utf-8") as f:
            items = ujson.load(f)
            for item in items:
                txt = f"{item['instruction']}{item['output']}{item['input']}"
                if len(txt) == 0 or len(txt) > 512:
                    continue
                lines.append(item)

    tb = pa.Table.from_pylist(lines)
    pq.write_table(
        table=tb,
        where=output_file,
        row_group_size=20480,
        data_page_size=20480,
    )


# Pretrain using WIKI and baidu baike
# gen_wiki_filter("../data/wikipedia-cn-20230720-filtered.json")
# 这里的563w_baidubaike要记得解压. 原本download的是7z压缩文件》
# gen_baike("/data/MINI_LLM_data/563w_baidubaike.json")
# gen_bell()  # To generate the eval dataset


# 原本的gen_sky 需要复制多个，没办法读取一个文件夹. 新的gen_sky只需要输入文件夹和输出文件夹的路径即可。并且原本的也会自动修改为.parquet结尾.（喵德注释）
# gen_sky_for_folder("/home/miaode/MINI_LLM/data/SkyPile-150B/data_folder","/home/miaode/MINI_LLM/datasets" )

# 这个在readme 没有说清楚是要下载哪一个记得是self_cognition.json 。（喵德注释)
# https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/data/self_cognition.json
# gen_aplca_sft("../data/alpaca_data_zh_51k.json","../datasets/aplca3.parquet")

# gen_bell_sft("../../../datasets/train_2M_CN.json", "../datasets/bell3.parquet")

# gen_mbvc("../datasets/oscar_202201.part_0000.jsonl","../datasets/mbvc1.parquet")

# gen_emotion_ds("../data/emotion.jsonl")
# gen_alpaca_train_sft("../datasets/sft.parquet")
gen_wiki("./raw_data/wikipedia-zh-cn-20240820.json", "./datasets/wiki-zh-cn.parquet")