import streamlit as st
import numpy as np
import pickle as pkl

best_model = pkl.dump(randomforest, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + randomforest.__class__.__name__ + '.pkl', 'wb'))

#function to predict the player rating using the best model
def predict_rating(data):
    predict = best_model.predict([data])
    return predict[0]

#streamlit app operations
st.title("Prediction of the Overall Rating of Players")

#defining the top features
top_features = ['overall', 'movement_reactions', 'mentality_composure', 'potential', 'release_clause_eur', 'wage_eur', 'value_eur', 'power_shot_power', 'passing', 'mentality_vision']

#taking new input from user
player_features = []
for f in top_features:
    value = st.number_input(f"Kindly enter a value for {f}:", min_value = 0.0, step = 0.1)
    player_features.append(value)

if st.button("Predict Player Rating"):
    player_rating = predict_rating(player_features)
    st.write(f"The predicted rating of the player is: {player_rating:.4f}")

st.success(player_features)

if __name__ == '__main__':
    main()

streamlit run s_app.py
    






