import pdfplumber
from pdfplumber.ctm import CTM
from tqdm import tqdm 
import copy 
import os
import uuid
import json
import logging

from llama_index import GPTVectorStoreIndex, VectorStoreIndex
from llama_index import SimpleDirectoryReader
from llama_index import Document

import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

logger = logging.getLogger(__name__)

# pdfplumber 활용해서 단어 추출
def extract_words(filename):
    pdf = pdfplumber.open(filename)
    pages = pdf.pages

    word_list_pages = []
    
    # 가로 형태의 pdf 처리를 위해
    width, height = pages[0].width, pages[0].height
    is_vertical = 1 if (height < width) else 0
    
    for i in tqdm(range(len(pages))):
        # 이걸 하면 page 에 있는 word 들이 하나하나 나오게 됩니다 dict 형태
        word_list_page = pages[i].extract_words(keep_blank_chars=False, use_text_flow=True, split_at_punctuation=True, expand_ligatures=True, extra_attrs=['fontname','size'])
        
        for j in word_list_page:
            # 가로 형태의 PDF이고 x0이 width의 중간 이상이면 1 -> 이후 is_split_page로 sorting
            j['is_split_page'] = 1 if (is_vertical == 1 and j['x0'] > width / 2) else 0
            j['bold'] = 1 if 'Bold' in j['fontname'] else 0
            j['page'] = i+1
            
            del j['x1']
            del j['upright']
            del j['direction']
            del j['top']
            del j['bottom']
            del j['fontname']
            
        word_list_page_sorted = sorted(word_list_page, key=lambda x : (x['is_split_page'], x['doctop'], x['x0'])) 
        word_list_pages += word_list_page_sorted
        
    return word_list_pages

# 같은 높이의 단어들을 연결해 줄을 구성
def extract_lines(input_list):
    results = []
    doctop_v = 0
    result = {}
    for i in range(len(input_list)):
        tmp = copy.deepcopy(input_list[i])
        text = tmp['text']
        
        if i == 0:
            doctop_v = tmp['doctop']
            result = tmp
            continue
        
        if (tmp['doctop'] != doctop_v):
            doctop_v = tmp['doctop']
            result['text'] = result['text'].replace(' . .','..').replace('..','')
            results.append(result)
            result = {}
            
        if len(result) == 0 :
            result = tmp
        else:
            result['text'] = result['text'] + ' ' + text
    return results

# 단락 구성
def extract_paragraphs(input_list, max_l):
    results = []
    
    prev_dict = input_list[0]
    result = {'par_text':''}
    for i in range(len(input_list)):
        tmp_dict = input_list[i]
        
        # (연결하려는 글자 길이가 2 초과) & (볼드체가 등장하거나 폰트 크기가 0.5 이상 커짐) & (현재 전체 텍스트 길이가 30 초과) 인 경우 단락 마무리
        if ((len(tmp_dict['text']) > 2) and ((prev_dict['bold'] == 0 and tmp_dict['bold'] == 1) or (prev_dict['size'] + 0.5 < tmp_dict['size'])) and len(result['par_text']) > 30):
            results.append(result)
            result = {'par_text':''}
        
        # 첫 라인을 기준으로 title, page 결정
        if len(result['par_text']) == 0:
            result['par_text'] = result['par_text']+ ' '
            result['par_title'] = tmp_dict['text']
            result['par_page'] = tmp_dict['page']
        else:
            result['par_text'] = result['par_text']+ ' ' + tmp_dict['text']
        
        prev_dict = input_list[i]
    results.append(result)
    
    # 글자수(max_l)로 자르기
    results2 = []
    for i in range(len(results)):
        tmp_dict = results[i]
        t = 1 if len(tmp_dict['par_text']) < (max_l * 2 + 1) else (len(tmp_dict['par_text'])-1) // max_l
        for j in range(t):
            tmp_dict2 = copy.deepcopy(tmp_dict)
            if j == t-1:
                tmp_dict2['par_text'] = tmp_dict['par_text'][max(0, (j*max_l - 10)):][:280]
            else:           
                tmp_dict2['par_text'] = tmp_dict['par_text'][max(0, (j*max_l - 10)):(j+1)*max_l]
            tmp_dict2['uuid'] = str(uuid.uuid4())
            tmp_dict2['par_content'] = tmp_dict2['par_title'][:40] + ' ' + tmp_dict2['par_text']
            results2.append(tmp_dict2)
        
    return results2

# 전체 sliding window
def extract_all(input_list, max_l):
    all_text = ''
    
    for i in range(len(input_list)):
        
        tmp_dict = input_list[i]
        all_text = all_text + ' ' + tmp_dict['text']
    
    results = []
    sub_text = ''
    for i in range(len(all_text)):
        sub_text = sub_text + all_text[i]
        
        if ((len(sub_text) > 400) and (all_text[i] == '.')) or (len(sub_text) > 500) :
            result = {}
            result['par_text'] = sub_text
            result['par_content'] = sub_text
            result['uuid'] = str(uuid.uuid4())
            result['par_page'] = 1
            result['par_title'] = 'test'
            results.append(result)
            sub_text = ''
        
    return results

def get_paragraphs(filename):
    words = extract_words(filename)
    lines = extract_lines(words)
    paragraphs = extract_paragraphs(lines, 230)
    return paragraphs

def get_all(filename):
    words = extract_words(filename)
    lines = extract_lines(words)
    results = extract_all(lines, 230)
    return results

def make_pdf2json(data_path, save_result_path = None):
    data_template = {
        "version": "Squad_Insurance",
        "data": []
    }

    document_template = {
        "uuid": "",
        "document_title": "",
        "document_company": "",
        "paragraphs": []
    }

    # json 추출
    document_json = copy.deepcopy(document_template)
    document_json['uuid'] = str(uuid.uuid4())
    document_json['document_title'] = data_path.rstrip('.pdf').split('/')[-1]
    document_json['document_company'] = 'None'
    document_json['paragraphs'] = get_all(os.path.join(data_path))
    
    data_template['data'].append(document_json)
                
    # 결과 json 파일 저장
    if save_result_path != None:
        with open(save_result_path, 'w', encoding="utf-8-sig") as file:
            json.dump(data_template, file, ensure_ascii=False, indent="\t")
            
    return data_template


def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    output_string = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    with open(pdf_path, 'rb') as file:
        interpreter = PDFPageInterpreter(resource_manager, TextConverter(resource_manager, output_string, codec=codec, laparams=laparams))
        for page in PDFPage.get_pages(file, check_extractable=True):
            interpreter.process_page(page)
            text = output_string.getvalue()
            yield text
            output_string.truncate(0)
            output_string.seek(0)
            
def text_postprocess(t):
    docs = []
    for page_number, page_text in enumerate(t,start=1):
        docs.append({'page_num' : page_number, 'text' : page_text})
    return docs

def make_pdf2json_llama(filepath):
    data_template = {
        "version": "Squad_Insurance",
        "data": []
    }

    document_template = {
        "uuid": "",
        "name": "",
        "content": "",
        "paragraphs": []
    }

    file_name = filepath.rstrip('.pdf').split('/')[-1]
    
    # json 추출
    document_json = copy.deepcopy(document_template)
    document_json['uuid'] = str(uuid.uuid5(uuid.NAMESPACE_DNS,file_name))
    document_json['name'] = file_name
    document_json['content'] = file_name
    
    # node 넣기
    documents = SimpleDirectoryReader(input_files=[filepath]).load_data()

    #text = extract_text_from_pdf(filepath)
    #ans = text_postprocess(text)
    #documents = [Document(t['text']) for t in ans]
    
    index = GPTVectorStoreIndex.from_documents(documents)
   
    doc_store = index.storage_context.to_dict()['doc_store']['docstore/data']
    vector_store = index.storage_context.to_dict()['vector_store']['embedding_dict']
    
    key_list = list(doc_store.keys())
    
    results = []
    for i in key_list:
        result = {}
        tmp_text = doc_store[i]['__data__']['text']
        result['name'] = tmp_text[:20]
        result['content'] = tmp_text
        result['uuid'] = str(uuid.uuid5(uuid.NAMESPACE_DNS, tmp_text[:20]))
        result['vector'] = vector_store[i]
        results.append(result)
       
    document_json['paragraphs'] = results
    data_template['data'].append(document_json)
    return data_template
