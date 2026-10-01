# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

可视化tokenizer。

Author: pankeyu
Date: 2023/06/01
"""
import os
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer


st.set_page_config(
    page_title="Tokenizer Viewer",
)

SAVED_PATH = 'tokenizers/saved'
CACHED_PATH = 'tokenizers/cached'


if 'current_tokenizer' not in st.session_state:
    st.session_state['current_tokenizer'] = None
    st.session_state['current_tokenizer_vocab'] = {}
    st.session_state['current_tokenizer_added_vocab'] = {}


if 'diff_tokenizer_1' not in st.session_state:
    st.session_state['diff_tokenizer_1'] = None
    st.session_state['diff_tokenizer_1_vocab'] = {}
    st.session_state['diff_tokenizer_1_added_vocab'] = {}
    st.session_state['diff_tokenizer_2'] = None
    st.session_state['diff_tokenizer_2_vocab'] = {}
    st.session_state['diff_tokenizer_2_added_vocab'] = {}


def load_tokenizer(
        token_path_or_name: str
    ):
    """
    加载指定的tokenizer。
    """
    return AutoTokenizer.from_pretrained(
        token_path_or_name,
        trust_remote_code=True
    )
    

def show_vocab_area(
        vocab: dict,
        vocab_df: pd.DataFrame,
        source: str
    ):
    """
    展示指定的vocab。

    Args:
        vocab (dict): _description_
    """
    search_token = st.text_input(
        f'在 {source} 中搜索指定 token',
        placeholder=f'在 {source} 中搜索指定 token...',
        label_visibility='collapsed'
    )

    if search_token:
        token_idx = (
            -1 
            if search_token not in vocab 
            else vocab[search_token]
        )

        if token_idx != -1:
            st.success(f'Token 「{search_token}」 存在于当前词表中，token_idx 为 {token_idx}。')
        else:
            st.warning(f'Token 「{search_token}」 不存在于当前词表中。')
    
    st.dataframe(
        vocab_df, 
        use_container_width=True,
        height=600
    )


def show_tokenize_area():
    """
    进行 tokenizer encode/decode 测试。
    """
    with st.expander('编/解码测试', expanded=True):
        text = st.text_input(
            '[>>>] Encoding Test: ',
            value='这是一个待编码的句子'
        )

        if text:
            ids = st.session_state['current_tokenizer'](text)['input_ids']
            tokens = st.session_state['current_tokenizer'].convert_ids_to_tokens(ids)
            st.code(f'encode token：{tokens}\nencode token ids：{ids}')
        
        id_list = st.text_input(
            '[<<<] Decoding Test: ',
            value='[0, 1, 2]'
        )

        if id_list:
            try:
                id_list = eval(id_list)
                tokens = st.session_state['current_tokenizer'].convert_ids_to_tokens(id_list)
                tokens2 = st.session_state['current_tokenizer'].decode(id_list)
                st.code(f'convert_ids_to_tokens 结果：{tokens}')
                st.code(f'decode 结果：{tokens2}')
            except Exception as e:
                st.error(e)


def get_vocab_dataframe(vocab: dict):
    """
    将vocab转换成df形式。

    Args:
        vocab (dict): _description_
    """
    temp_dict = {
        'token': list(vocab.keys()),
        'idx': list(vocab.values())
    }
    return pd.DataFrame.from_dict(temp_dict)


def sort_vocab_by_index(vocab_dict: dict):
    """
    将 vocab_dict 按照 index 进行排序。
    """
    return dict(sorted(vocab_dict.items(), key=lambda x: x[1]))


def start_viewer_page():
    """
    可视化tokenizer。
    """
    st.code("""该页面用于可视化 tokenizer 信息，以帮助更快速，更直观的了解 tokenizer，包含以下功能：
1. 查看 tokenizer 包含字符数（原始字符数、额外扩展字符数）。
2. 搜索 tokenizer 中是否包含指定 token。
3. encode 和 decode 功能测试。
    """)

    st.write('tokenizer 路径（huggingface 或 开发机 目录）:')

    c1, c2 = st.columns([10, 4])
    with c1:
        token_path_or_name = st.text_input(
            'tokenizer 路径（huggingface 或 开发机 目录）:',
            value=os.environ.get('VIEWER_TOKENIZER', 'bert-base-chinese'),
            label_visibility='collapsed'
        )
    
    with c2:
        btn = st.button('Load Tokenizer!')

    if btn:
        with st.spinner('Loading tokenizer...'):
            st.session_state['current_tokenizer'] = load_tokenizer(token_path_or_name)
            st.session_state['current_tokenizer_vocab'] = sort_vocab_by_index(st.session_state['current_tokenizer'].get_vocab())
            st.session_state['current_tokenizer_vocab_list'] = list(st.session_state['current_tokenizer_vocab'].keys())
            st.session_state['current_tokenizer_vocab_df'] = get_vocab_dataframe(st.session_state['current_tokenizer_vocab'])
            st.session_state['current_tokenizer_added_vocab'] = sort_vocab_by_index(st.session_state['current_tokenizer'].get_added_vocab())
            st.session_state['current_tokenizer_added_vocab_list'] = list(st.session_state['current_tokenizer_added_vocab'].keys())
            st.session_state['current_tokenizer_added_vocab_df'] = get_vocab_dataframe(st.session_state['current_tokenizer_added_vocab'])

    if st.session_state['current_tokenizer']:
        with st.expander('字符数统计', expanded=True):
            temp_dict = {
                'tokenizer': [token_path_or_name],
                'vocab_len': [len(st.session_state['current_tokenizer_vocab']) - len(st.session_state['current_tokenizer_added_vocab'])],
                'added_vocab_len': [len(st.session_state['current_tokenizer_added_vocab'])],
                'all_vocab_len': [len(st.session_state['current_tokenizer_vocab'])]
            }
            df = pd.DataFrame.from_dict(temp_dict)
            st.dataframe(df, use_container_width=True)
    
        c1, c2 = st.columns([5, 5])
        
        with c1:
            show_vocab_area(
                st.session_state['current_tokenizer_vocab'],
                st.session_state['current_tokenizer_vocab_df'],
                source='vocab'
            )

        with c2:
            show_vocab_area(
                st.session_state['current_tokenizer_added_vocab'],
                st.session_state['current_tokenizer_added_vocab_df'],
                source='added vocab'
            )

        show_tokenize_area()


def show_diff_area():
    """
    差分出两个tokenizer不同的token。
    """
    c1, c2 = st.columns([5, 5])

    tokenizer1_unique_token = set(st.session_state['diff_tokenizer_1_vocab_list']) - set(st.session_state['diff_tokenizer_2_vocab_list'])
    tokenizer2_unique_token = set(st.session_state['diff_tokenizer_2_vocab_list']) - set(st.session_state['diff_tokenizer_1_vocab_list'])

    with c1:
        st.write(f'tokenizer 1 独有的 tokens（{len(tokenizer1_unique_token)}）：')
        temp_dict = {
            'more tokens': list(tokenizer1_unique_token)
        }
        st.dataframe(
            temp_dict,
            use_container_width=True,
            height=600
        )
    
    with c2:
        st.write(f'tokenizer 2 独有的 tokens（{len(tokenizer2_unique_token)}）：')
        temp_dict = {
            'more tokens': list(tokenizer2_unique_token)
        }
        st.dataframe(
            temp_dict,
            use_container_width=True,
            height=600
        )


def start_diff_page():
    """
    用于比较两个tokenizer的不同。
    """
    st.code("""该页面用于比较 2 个 tokenizer 之间的 diff token，用于了解 tokenizer 之间的差异：
1. 查看 tokenizer 之间有哪些 tokens 不同。
2. 将 2 个tokenizer 之间做 merge。
    """)

    c1, c2, c3 = st.columns([10, 10, 6])
    with c1:
        st.write('tokenizer 1 路径（huggingface 或 开发机 目录）:')
    with c2:
        st.write('tokenizer 2 路径（huggingface 或 开发机 目录）:')

    c1, c2, c3 = st.columns([10, 10, 6])
    with c1:
        token_1_path_or_name = st.text_input(
            'tokenizer 1 路径（huggingface 或 开发机 目录）:',
            value='bert-base-chinese',
            label_visibility='collapsed'
        )
    with c2:
        token_2_path_or_name = st.text_input(
            'tokenizer 2 路径（huggingface 或 开发机 目录）:',
            value='tiiuae/falcon-7b',
            label_visibility='collapsed'
        )
    with c3:
        start_btn = st.button(
            'Load！'
        )

    if start_btn:

        if token_1_path_or_name and token_2_path_or_name:
            with st.spinner('Loading tokenizer...'):
                st.session_state['diff_tokenizer_1'] = load_tokenizer(token_1_path_or_name)
                st.session_state['diff_tokenizer_1_vocab'] = sort_vocab_by_index(st.session_state['diff_tokenizer_1'].get_vocab())
                st.session_state['diff_tokenizer_1_vocab_list'] = list(st.session_state['diff_tokenizer_1_vocab'].keys())
                st.session_state['diff_tokenizer_1_vocab_df'] = get_vocab_dataframe(st.session_state['diff_tokenizer_1_vocab'])
                st.session_state['diff_tokenizer_1_added_vocab'] = sort_vocab_by_index(st.session_state['diff_tokenizer_1'].get_added_vocab())
                st.session_state['diff_tokenizer_1_added_vocab_list'] = list(st.session_state['diff_tokenizer_1_added_vocab'].keys())
                st.session_state['diff_tokenizer_1_added_vocab_df'] = get_vocab_dataframe(st.session_state['diff_tokenizer_1_added_vocab'])
                
                st.session_state['diff_tokenizer_2'] = load_tokenizer(token_2_path_or_name)
                st.session_state['diff_tokenizer_2_vocab'] = sort_vocab_by_index(st.session_state['diff_tokenizer_2'].get_vocab())
                st.session_state['diff_tokenizer_2_vocab_list'] = list(st.session_state['diff_tokenizer_2_vocab'].keys())
                st.session_state['diff_tokenizer_2_vocab_df'] = get_vocab_dataframe(st.session_state['diff_tokenizer_2_vocab'])
                st.session_state['diff_tokenizer_2_added_vocab'] = sort_vocab_by_index(st.session_state['diff_tokenizer_2'].get_added_vocab())
                st.session_state['diff_tokenizer_2_added_vocab_list'] = list(st.session_state['diff_tokenizer_2_added_vocab'].keys())
                st.session_state['diff_tokenizer_2_added_vocab_df'] = get_vocab_dataframe(st.session_state['diff_tokenizer_2_added_vocab'])
        else:
            st.warning('请同时填写两个路径！')
    
    if st.session_state['diff_tokenizer_1']:
        with st.expander('字符数统计', expanded=True):
            temp_dict = {
                'tokenizer': [
                    token_1_path_or_name, 
                    token_2_path_or_name,
                    'diff'
                ],
                'vocab_len': [
                    len(st.session_state['diff_tokenizer_1_vocab']) - len(st.session_state['diff_tokenizer_1_added_vocab']),
                    len(st.session_state['diff_tokenizer_2_vocab']) - len(st.session_state['diff_tokenizer_2_added_vocab']),
                    len(st.session_state['diff_tokenizer_1_vocab']) - len(st.session_state['diff_tokenizer_1_added_vocab']) \
                        - len(st.session_state['diff_tokenizer_2_vocab']) + len(st.session_state['diff_tokenizer_2_added_vocab'])
                ],
                'added_vocab_len': [
                    len(st.session_state['diff_tokenizer_1_added_vocab']),
                    len(st.session_state['diff_tokenizer_2_added_vocab']),
                    len(st.session_state['diff_tokenizer_1_added_vocab']) - len(st.session_state['diff_tokenizer_2_added_vocab'])
                ],
                'all_vocab_len': [
                    len(st.session_state['diff_tokenizer_1_vocab']),
                    len(st.session_state['diff_tokenizer_2_vocab']),
                    len(st.session_state['diff_tokenizer_1_vocab']) - len(st.session_state['diff_tokenizer_2_vocab'])

                ]
            }
            df = pd.DataFrame.from_dict(temp_dict)
            st.dataframe(df, use_container_width=True)
        
        show_diff_area()


def main():
    """
    主函数流程。
    """
    st.markdown(
        "<h1 style='text-align: center;'>🤗 Tokenizer Viewer</h1>", 
        unsafe_allow_html=True
    )
    
    viewer_page, merge_page, train_page = st.tabs([
        'viewer_page',
        'diff_page',
        'train_page'
    ])
    
    with viewer_page:
        start_viewer_page()

    with merge_page:
        start_diff_page()

    with train_page:
        pass


if __name__ == '__main__':
    main()