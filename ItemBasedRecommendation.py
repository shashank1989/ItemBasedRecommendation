import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pandasql import sqldf
from sklearn.model_selection import train_test_split
from pandasql import sqldf
import joblib
from flask import Flask, request, jsonify, render_template, request
import itertools
app = Flask(__name__)

def User_Recommendation(items_data,dictionary,sims,itemid,top_recommendation):
    # Calculate the Item Based Similarity for the given Item 
    word_list_item_recommendation = items_data['Specification'][items_data['ItemID'] == itemid].iloc[0].lower().split(',')
    #print(word_list_item_recommendation)
    #print("Since Student liked item {} please find below recommendation".format(itemid))
    query_doc = word_list_item_recommendation
    query_doc_bow = dictionary.doc2bow(query_doc)
    similarity_score = sims[query_doc_bow]
    SimilarityDf = pd.DataFrame(similarity_score.tolist(),index=items_data.ItemID.values,columns=['Score'])
    SimilarityDf = SimilarityDf['Score'].astype(float)
    SimilarityDf = SimilarityDf.reset_index().rename(columns={'index': 'ItemID'})
    query = f"select ItemID from SimilarityDf where Score > .5 order by Score desc limit {top_recommendation}"
    result_list = sqldf(query).values.tolist()
    # Flatten the list of lists to a single list
    flattened_list = list(itertools.chain.from_iterable(result_list))
    return flattened_list

# Load the model from the file 
sims_from_joblib = joblib.load('similarity.pkl') 
dictionary_from_joblib = joblib.load('dictionary.pkl') 
items_data_joblib = joblib.load('items_data.pkl') 

def recommend(student_id,item_id):
    recommendation = User_Recommendation(items_data_joblib,dictionary_from_joblib,sims_from_joblib,int(item_id),10)
    recomm = f"Student ID : {student_id} , ITEM_ID : {item_id} are {recommendation}"
    return recomm

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        student_id = request.form['student_id']
        item_id = request.form['item_id']
        
        # Call recommend function with student_id and item_id
        recommendation = recommend(student_id, item_id)
        
        return render_template('Recommendation.html', recommendation=recommendation)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)




