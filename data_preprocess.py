
import re 
import webvtt
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
import json
    

def preprocess_data(dataset,path):
    
    stac=[]
    start_timestamp=[]
    dataset=pd.read_csv(dataset)
    start_timestamp.append(dataset['start_timestamp'][0])
    end_timestamp=[]
    start_time=[]
    st=[]
    a=len(dataset)
    for i in range(len(dataset)-1):
        if dataset['end_timestamp'][i][3:5]!=dataset['end_timestamp'][i+1][3:5]:
            d=list(dataset.loc[:i,'text'])
            listToStr = ' '.join([str(elem) for elem in d])
            start_timestamps=dataset['start_timestamp'][i+1]
            end_timestamps=dataset['end_timestamp'][i]
            stac.append(listToStr)
            start_timestamp.append(start_timestamps)
            end_timestamp.append(end_timestamps)
        elif  dataset['end_timestamp'][len(dataset['end_timestamp'])-1][0:5]==dataset['end_timestamp'][i][0:5]:
            d=dataset['text'][i]
            #listToStr = ' '.join([str(elem) for elem in d])
            start_times=dataset['start_timestamp'][i]
            st.append(d)
            start_time.append(start_times)
    
    
    # joining context for few last timestamps as it does not fall in one minute interval
    
    st = ' '.join([str(elem) for elem in st])
    start_timestamp=start_timestamp[:-1]
    start_timestamp.append(start_time[1])
    end_timestamp.append(dataset['end_timestamp'][len(dataset['end_timestamp'])-1])
    
    tag=pd.read_excel('{}/tags.xlsx'.format(path))


    context=[]
    for i in range(len(stac)):
        if i==0:
            stacs=stac[0]
        else:
            stacs=stac[i][len(stac[i-1]):]
        context.append(stacs)
    context.append(st)
    
    df = pd.DataFrame(list(zip(start_timestamp,end_timestamp,context)),
               columns =['start_timestamp', 'end_timestamp','context'])
    
    names = context

    with open("C:/Users/Vijaya/Topic_modelling_project/stop.txt","r") as sw:
        stop_words = sw.read()
    
    stop_words = stop_words.split("\n")
    stop_words.append('yeah')
    
    lemmatizer = WordNetLemmatizer()
    # Removing unwanted symbols incase if exists
    index=names.index(names[-1])
    ip_rev_strings=[]
    for i in range(index+1):
        ip_rev_string = re.sub("[^A-Za-z" "]+"," ", names[i]).lower()
        ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)
        ip_rev_string=lemmatizer.lemmatize(ip_rev_string)
        ip_reviews_word = ip_rev_string.split(" ")
        ip_reviews_word = [w for w in ip_reviews_word if not w in stop_words]
        ip_rev_string = " ".join(ip_reviews_word)
        ip_rev_strings.append(ip_rev_string)
    
    model = SentenceTransformer('paraphrase-distilroberta-base-v2')
    candidate_embeddings = model.encode(tag['Description'])
    from sklearn.metrics.pairwise import cosine_similarity
    suit=[]
    link=[]
    for i in range(len(ip_rev_strings)):
        doc_embedding = model.encode([ip_rev_strings[i]])
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        dist=distances[0][np.where(distances[0]>0.3)]
        dist=dist.tolist()
        dist.sort(reverse=True)
        if len(dist)==2:
            tap=list(tag['Topic'][distances[0]==dist[0]]),list(tag['Topic'][distances[0]==dist[1]])
            lin=list(tag['Link'][distances[0]==dist[0]]),list(tag['Link'][distances[0]==dist[1]])
        elif len(dist)>2:
            tap=list(tag['Topic'][distances[0]==dist[0]]),list(tag['Topic'][distances[0]==dist[1]]),list(tag['Topic'][distances[0]==dist[2]])
            lin=list(tag['Link'][distances[0]==dist[0]]),list(tag['Link'][distances[0]==dist[1]]),list(tag['Link'][distances[0]==dist[2]])
        elif len(dist)==0:
            tap=[]
            lin=[]
        elif len(dist)==1:
            tap=list(tag['Topic'][distances[0]==dist[0]])
            lin=list(tag['Link'][distances[0]==dist[0]])
        tap=list(tap)
        suit.append(tap)
        link.append(lin)
        
    #remove bracket from Topics
    
    for i in range(len(suit)):
        suit[i]=str(suit[i]).replace('[','').replace(']','')
    
    #remove bracket from link    
    
    for i in range(len(link)):
        link[i]=str(link[i]).replace('[','').replace(']','').replace('(','').replace(')','')
    
       
    #Adding Topic column to dataframe
    
    df['Topics']=suit
    
    #Adding Link column to dataframe
    
    df['Link']=link
     
# Serializing json  
    json_object = json.dumps(df.to_dict('list'), indent = 4)
    with open("C:/Users/Vijaya/Topic_modelling_project/uploads/topic.json","w") as js_ob:
        js_ob.write(json_object)
    
    return json_object,df
    



def convert_to_vtt(path,UPLOAD_FOLDER):
    #path='C:/Users/Vijaya/Topic_modelling_project/'
    
    #caption_path = "{}/Session18_26th_dec2021.vtt".format(path)
    start_time = []
    end_time = []
    text = []
    for caption in webvtt.read(path):
        start_time.append(caption.start.split(".")[0])
        end_time.append(caption.end.split(".")[0])
        try:
            text.append(caption.text.split(":")[1])
        except:
            text.append(caption.text)
    
    #creating a dataframe 
    
    dataset = pd.DataFrame()
    dataset["start_timestamp"] = start_time
    dataset["end_timestamp"] = end_time
    dataset["text"] = text
    
    #saving the csv file
    dataset.to_csv("{}/vtt_df.csv".format(UPLOAD_FOLDER), index=False)
    return dataset