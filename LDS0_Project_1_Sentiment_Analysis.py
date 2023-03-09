import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import underthesea
from underthesea import word_tokenize
import warnings
import re
import emoji
import time
import string
import pickle

from sklearn.model_selection import cross_val_score
import datetime
from sklearn.metrics import classification_report

#------------------
# Hiệu ứng
st.balloons()

#------------------
# Backends Functions Processing: 
# Function to load file dict
def read_textfiles(file,encode):
    with open(file, 'r', encoding=encode) as file:
        lst_text = file.read()
    lst_text = lst_text.split('\n')
    file.close()
    return lst_text

def convert_textfiles(x):
    files=open(x,'r',encoding='utf8')
    lines=files.read().splitlines()
    dict_={}
    for i in lines:
        key,value=i.split('\t')
        dict_[key]=str(value)
    files.close()
    return dict_

# Dictionary
teen_dict       = convert_textfiles('files/teencode.txt') 
EV_dict         = convert_textfiles('files/english-vnmese.txt')
stop_words      = read_textfiles('files/stopwords.txt','utf-8')
wrong_text_list = read_textfiles('files/wrong-word.txt','utf-8')
emo_lst         = read_textfiles('files/emojicon.txt','utf-16')

### Sub Function:
def label_rate(rating):
    if rating > 4:
        return 'Good'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Very Bad'

## Function remove html
def remove_html(text): # loại bỏ tag HTML
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# Function remove các ký tự trùng: vd: xấuuuuuuuuuu
def dupe_characters_remove(text):
    return re.sub(r'([A-Z])\1+', lambda m: m.group(1), text, flags=re.IGNORECASE)

# Function to replace teen code
def teencode_replace(text):
    for key, value in teen_dict.items():
        text= re.sub(rf'\b{key}\b', value, text)
    return text

# Function to replace english-vietnamese
def EV_replace(text):
    for key, value in EV_dict.items():
        text= re.sub(rf'\b{key}\b', value, text)
    return text

# Function to emoticon bằng text
def replace_emoji(text):
    result = " "
    for char in text:
        if emo_lst(char):
            result = result +" "+ emo_lst.get(char, char)
        else:
            result += emo_lst.get(char, char)
    return result

# Function to PRE-PROCESS TEXT
def chuan_hoa_vanban(text):
    df = pd.DataFrame(text,columns=['text'])
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'\n',' ',regex=True)
    df['text'] = df['text'].apply(lambda text: remove_html(text))
    df['text'] = df['text'].apply(lambda text: dupe_characters_remove(text))
    df['text'] = df['text'].apply(lambda text: teencode_replace(text))
    df['text'] = df['text'].apply(lambda text: EV_replace(text))
    df['text'] = df['text'].apply(lambda text: replace_emoji(text))
    #word_tokenize
    df['text'] = df['text'].apply(lambda text: word_tokenize(text, format='text'))
    
 # Function to Split sentence
    df['text_words'] = [[text for text in x.split()] for x in df['text']]
    df['text_words'] = [[re.sub('[0-9]+','', e) for e in text] for text in df['text_words']] # remove number
    df['text_words'] = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '(', ')', '+', '/','!','@','#','$','%']] for text in df['text_words']] # remove special characters
    df['text_words'] = [[t for t in text if not t in wrong_text_list] for text in df['text_words']] # wrong_word remove
    df['text_words'] = [[t for t in text if not t in stop_words] for text in df['text_words']] # stopword remove
    df['text_words'] = df['text_words'].map(' '.join)
    return df['text_words']   

# Function to change results to text
def ketqua(i):  
    if i ==1:
        return 'Like'
    if i==-1:
        return 'Not Like'
    return 'Bình thường'

# Function to read data
def run_all(df):
    # 1. Show raw data
    st.subheader('1. Thông tin bộ dữ liệu dùng để huấn luyện mô hình')
    st.write('5 dòng dữ liệu đầu')
    st.dataframe(df.head(5))
    st.write('5 dòng dữ liệu cuối')
    st.dataframe(df.tail(5))
    st.write('Kích thước dữ liệu')
    st.code('Số dòng: '+str(df.shape[0]) + ' và số cột: '+ str(df.shape[1]))
    n_null = df.isnull().any().sum()
    st.code('Số dòng bị NaN: '+ str(n_null))

    st.subheader('Thống kê dữ liệu')
    st.dataframe(raw_df.describe())
    st.write('Dữ liệu có điểm đánh giá (rating_score) là phù hợp, với min là 1 và max là 5, trung bình là 3 điểm')

######LOAD MODEL ĐÃ BUILD 

with open('model/Shopee_review_Logistic.pkl', 'rb') as model_file:  #### model này có 90% accuracy nên khá tối ưu
    phanloai_review = pickle.load(model_file)
    
#------------------------------------------------
# GUI
menu = ["Business Objective Overview", "Results of Build Sentiment Analysis Model", "Predict new customers"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective Overview':
    st.markdown("<h1 style='text-align: center; color: black;'>Sentinent Analysis</h1>", unsafe_allow_html=True)   
    st.subheader("Objective Overview")
    st.write(""" Dựa trên dữ liệu review Thời trang nam bằng tiếng Việt thu thập từ Shopee, 
    thực hiện lựa chọn thuật toán Machine Learning và xây dựng công cụ nhằm tự động
    phân loại phản hồi khách hàng thành 3 loại:
    Tích cực(Positive), Tiêu cực(Negative) và Trung tính(Neutral)""")  
    st.write("""###### => Segment/group/cluster of customers (market segmentation is also known as market segmentation)
     is the process of grouping or collecting customers together based on common characteristics. It divides and groups
     customers into subgroups according to psychological, behavioral.
             """)
    st.image("Shopee.png")
    st.image("sentiment1.png")
    st.write("""#### Phân tích Sentiment Analysis: là một kĩ thuật phân khúc 
            dựa trên hành vi đánh giá của KH mà chúng ta biết được KH có hài lòng với SP or Service hay không.""")
   
    st.header('Sentiment Analysis là gì ?')
    st.markdown('''
    Sentiment analysis phân tích tình cảm (hay còn gọi là phân tích quan điểm phân tích
    cảm xúc phân tính cảm tính là cách sử dụng xử lý ngôn ngữ tự nhiên phân tích
    văn bản ngôn ngữ học tính toán , và sinh trắc học để nhận diện, trích xuất,
    định lượng và nghiên cứu các trạng thái tình cảm mà thông tin chủ quan
    một cách có hệ thống. 
    
    Sentiment analysis được áp dụng rộng rãi cho các tài liệu chẳng hạn như các đánh giá
    và các phản hồi khảo sát, phương tiện truyền thông xã hội, phương tiện truyền thông
    trực tuyến, và các tài liệu cho các ứng dụng từ marketing đến quản lý quan hệ
    khách hàng và y học lâm sàng.

    Điều này trở nên quan trọng hơn trong ngành kinh doanh Thời Trang. Các Shop
    cần nỗ lực để cải thiện chất lượng của Product and Service cũng như thái độ phục vụ nhằm duy trì
    uy tín của Shop cũng như tìm kiếm thêm khách hàng mới.''')

elif choice == 'Results of Build Sentiment Analysis Model':
    st.subheader("Results Of Shopee Project")
    st.markdown("<h1 style='text-align: center; color: black;'>Capstone Project</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Shopee Sentiment Analysis</h2>", unsafe_allow_html=True)   
    st.image('category1.png')
    st.write('Sau khi Upload Shopee Data Review file chúng ta cần lựa chọn 1 loại sản phẩm phù hợp để dự đoán')
    st.image('rating_score.png')
    st.write('Tỷ lệ cho thấy rõ Tỷ lệ Rating score 5 rất cao nhưng vẫn có nhiều điểm chưa tốt mà các Shop cần nhận ra ngay')
    st.image('Logistic1.png')
    st.write('Sau khi xử lý data bỏ tất cả các từ meanless và dùng Logistic Regression để dự đoán ta có được data như hình')
    st.image('Logistic 2.png')
    st.write('Logistic Regession Model cho kết quả rất tốt ở mức 90%')
    st.image('Naive_Bayer1.png')
    st.write('Sau khi dùng Logictic Model, chúng ta cần 1 mo hình khác để so sánh')
    st.image('Naive 2.png')
    st.write('Tuy nhiên tỷ lệ của Naive_bayers model chưa tối ưu chỉ được 80%')
    
elif choice == 'Predict new customers':
    st.header('Choose Your Data Upload')
    flag = False
    lines=None
    type= st.radio('Upload file csv|txt hoặc Nhập liệu thủ công',options=('Upload file csv','Nhập nhận xét'))
    
    if type=='Upload file csv':
        upload_file = st.file_uploader('Chọn file',type =['txt','csv'])
        if upload_file is not None:  
            df = pd.read_csv(upload_file, encoding='utf-8')
            run_all(df)
            lines=pd.read_csv(upload_file,sep='\n',header=None,encoding='utf-8')   
            lines=lines.apply(' '.join)
            lines=np.array(lines)
            st.write('Nội dung review:')
            st.write((lines))
            flag=True
        else:
            st.write('Hãy upload file vào app')
    if type =='Nhập nhận xét':
        form_nhanxet = st.form("surprise_form")
        txt_noi_dung = form_nhanxet.text_area('Nhập nhận xét ở đây')
        bt_sur_submit = form_nhanxet.form_submit_button("Đánh giá")

        if bt_sur_submit:
            lines = np.array([txt_noi_dung])
            flag = True
    if flag == True:
        st.write('Phân loại nhận xét')
        if len(lines)>0:            
            processed_lines = chuan_hoa_vanban(lines)
            st.write('Văn bản sau khi xử lý')
            st.code((processed_lines).to_list())
            y_pred_new = phanloai_review.predict(processed_lines)    
            st.write('Kết quả phân loại cảm xúc')
            st.code(str(ketqua(y_pred_new)))
    

