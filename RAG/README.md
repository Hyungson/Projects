- text parshing
  
pdf -> pdfplumber

docx -> python-docx 사용해서 text 파싱


- preprocessing pipeline

pdf로 문서 통일 -> layout detection -> title, section header, content, page 등의 메타데이터와 청크 생성

   
layout detection은 yolo v8 사용

layout detection 파인튜닝을 위한 학습데이터 구축은 cvat 사용

nuclio로 모델 deploy, auto labeling 가능

section header, title 등은 문서 유형 별로 rule based code로 추가 고도화 필요

Langchain Chunking Algorithm은 md포멧 사용
