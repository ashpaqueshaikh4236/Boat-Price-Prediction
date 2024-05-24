import streamlit as st
import numpy as np
import pickle


# Load the dataset and model
df = pickle.load(open('clean_data.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title('Boat Price Prediction')


categorical = df.select_dtypes(include='object')
numerical = df.select_dtypes(exclude='object')


unique_values_dict = {}
for col in df.columns:
    unique_values = df[col].unique().tolist()
    unique_values.sort()
    unique_values.insert(0, 'select')
    unique_values_dict[col] = unique_values


join_data = []
for col, values in unique_values_dict.items():
    if col in categorical:
        come_cat_values= st.selectbox(col, values)
        join_data.append(come_cat_values)
    elif col in numerical:
         come_num_values = st.text_input(col)
         join_data.append(come_num_values) 

try:
    if st.button('predict'):
        reshaped_data = np.asarray(join_data).reshape(1,-1)
        prediction = model.predict(reshaped_data)
        st.success(prediction[0])
except:
    st.warning('Please Fill all values')
