import pdfplumber
from glob import glob
from tqdm import tqdm 
import copy 
import json
import re


pdf_path = glob('./data/out_law_pdf/금융소비자 보호에 관한 감독규정(금융위원회고시)(제2023-38호)(20230705).pdf')
pattern1 = r'[가-힣]\.'
pattern2 = r'\d\.|[\u2460-\u2473]'

# pdfplumber 활용해서 단어 추출
def extract_words(filename):
    pdf = pdfplumber.open(filename)
    pages = pdf.pages

    word_list_pages = []
    
    for i in tqdm(range(len(pages))):
        # 이걸 하면 page 에 있는 word 들이 하나하나 나오게 됩니다 dict 형태
        word_list_page = pages[i].extract_words(keep_blank_chars=False, use_text_flow=True, split_at_punctuation=True, expand_ligatures=True, extra_attrs=['fontname','size'])
        
        for j in word_list_page:
            # 가로 형태의 PDF이고 x0이 width의 중간 이상이면 1 -> 이후 is_split_page로 sorting
           
            j['page'] = i+1
            j['x0'] = int(j['x0'])
            j['x1'] = int(j['x1'])
            j['doctop'] = int(j['doctop'])
            
            #del j['x1']
            del j['upright']
            del j['direction']
            del j['top']
            del j['bottom']
            del j['fontname']
            
        word_list_page_sorted = sorted(word_list_page, key=lambda x : (x['doctop'], x['x0'])) 
        word_list_pages += word_list_page_sorted
        
    return word_list_pages


# 같은 높이의 단어들을 연결해 줄을 구성
def extract_lines(input_list):
    results = []
    doctop_v = 0
    result = {}
    high_text, low_text = [],[]
    prev = input_list[0]
    for i in range(len(input_list)):
        tmp = copy.deepcopy(input_list[i])

        text = tmp['text']
        
        if i == 0:
            doctop_v = tmp['doctop']
            result = tmp
            continue
        
        if (tmp['doctop'] != doctop_v):
            # test
            if ((542.0 <= prev['x1'] <= 566.0) and ((tmp['doctop'] - prev['doctop']) <= 20.0)) :
                if prev['text'][-1] == '.' or prev['text'][-1].isdigit() :
                    result['line_finish'] = 1
                else:
                    result['line_finish'] = 0

            elif tmp['text']=='제1장':
                result['line_finish'] = 1
                    
            else:
                result['line_finish'] = 1
          
            #

            doctop_v = tmp['doctop']
            result['text'] = result['text'].replace(' . .','..').replace('..','')
            results.append(result)
            result = {}
            
        if len(result) == 0 :
            result = tmp
        else:
            result['text'] = result['text'] + ' ' + text

        prev = input_list[i]
          
        result['x1'] = tmp['x1']
    # 워터마크 : 반복되는 페이지 맨위, 맨아래 line 삭제

    prev_dict = results[0]
    high_text.append(prev_dict)
    for i in range(len(results)):
        
        tmp = copy.deepcopy(results[i])
        if tmp['page'] > prev_dict['page']:
            low_text.append(prev_dict)
            high_text.append(tmp)
        
        prev_dict = results[i]
    
    for i in range(len(low_text)):
        results.remove(low_text[i])
    for i in range(len(high_text)):
        results.remove(high_text[i])


    ## 끝나지 않은 line을 한 문장으로 잇기
    results2=[]
    tmp_dict={'text':''}
    #prev_line = results[0]
    for i in range(len(results)):
        tmp_line = results[i]
        if tmp_line['line_finish'] == 0:
            if tmp_dict['text'] == '':
                tmp_dict = tmp_line
                tmp_dict['start_doctop'] = tmp_dict['doctop']
                del tmp_dict['doctop']
            else:
                tmp_dict['text'] = tmp_dict['text'] + ' ' + tmp_line['text']
                tmp_dict['x1'] = tmp_line['x1']
        else:
            if tmp_dict['text'] == '':
                tmp_line['start_doctop'] = tmp_line['doctop']
                tmp_line['end_doctop'] = tmp_line['doctop']
                del tmp_line['doctop']
                results2.append(tmp_line)
            else:
                tmp_dict['text'] = tmp_dict['text'] + ' ' + tmp_line['text']
                tmp_dict['end_doctop'] = tmp_line['doctop']
                tmp_dict['x1'] = tmp_line['x1']
                results2.append(tmp_dict)
                tmp_dict={'text':''}
            
    
    results3 = []
    for i in range(len(results2)):
        tmp = results2[i]
        if tmp['text'].split(' ')[0].startswith('제'):
            if tmp['text'].split(' ')[0].endswith('편'):
                tmp['title'] = '제n편'
            
            elif tmp['text'].split(' ')[0].endswith('장'):
                tmp['title'] = '제n장'

            elif tmp['text'].split(' ')[0].endswith('절'):
                tmp['title'] = '제n절'

            else: 
                tmp['title'] = '제n조'

        else:
            if i == 0:
                tmp['title'] = '대제목'

            else:
                string = tmp['text'].split(' ')[0]
                if re.match(pattern1, string):
                    tmp['title'] = '본문_가나다'

                elif re.match(pattern2, string):
                    tmp['title'] = '본문_숫자'

                else:
                    tmp['title'] = '본문_기타'

        results3.append(tmp)

    return results3

def pdf2json():
    for file_path in pdf_path:

        # odt 파일 경로와 저장할 JSON 파일 경로 지정
        pdf_file_path = file_path
        json_file_path = file_path.replace('.pdf', '.json')

        words_list = extract_words(pdf_file_path)
        lines_list = extract_lines(words_list)
        content_list=[]
        for line in lines_list:
            content_list.append(line)

        # print(cleaned_list)
        with open(json_file_path, 'w', encoding='utf-8') as json_output:
            json.dump({'content': content_list}, json_output, ensure_ascii=False, indent=4)
        
        print(f'{pdf_file_path} 파일을 {json_file_path}로 변환 및 저장했습니다.')

if __name__ == '__main__':
    pdf2json()