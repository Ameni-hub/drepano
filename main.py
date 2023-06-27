# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
import pandas as pd
df = pd.read_spss('C:/Users/Ameni/projet5.sav')
df5 = df
df["CDD"] = df["CDD"].astype(str)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import os


#st.title('_:red[Prediction of Splenic Sequestration in Sickle Cell Patients using AI at the National Center for Bone Marrow Transplantation of Tunisia]_')
st.markdown(
    "<h1 style='color: red; font-size: 32px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); text-align: center;'>Prediction of Splenic Sequestration in Sickle Cell Patients Using AI at the National Center for Bone Marrow Transplantation of Tunisia</h1>",
    unsafe_allow_html=True)
#image_path = 'C:/Users/Ameni/PycharmProjects/drepano/centre.jpg'  # Replace with the path to your image file


# Load and process the image
image_path = 'C:/Users/Ameni/PycharmProjects/drepano/centre.jpg'  # Replace with the path to your image file


# Create a sidebar with menu options
rad = st.sidebar.radio("**Menu**", ['Home', 'Aim of study', 'My Data', 'Prediction'])


if rad == "Home":
    col1, col2 = st.columns([2, 3])

    # Add the image to the left column
    with col1:
        image = Image.open('C:/Users/Ameni/PycharmProjects/drepano/centre.jpg')
        st.image(image, use_column_width=True)

        # Adjust the height and width of the image using CSS
        st.markdown(
            f"""
                <style>
                .st-ax {{ max-width: 100%; max-height: 600px; }}
                </style>
                """,
            unsafe_allow_html=True
        )

    # Add the content to the right column
    with col2:
        st.subheader("**:green[Bone Marrow Transplant Center]**")
        st.write(
            "The bone marrow transplant center was established in 1998. It specializes in all types of bone marrow transplantation and the treatment of individuals with immunodeficiency.")
        st.write("<span style='font-size:20px;'>**Departments:**</span>", unsafe_allow_html=True)

        st.write("**:black[Department 1]**")
        st.write("The first department is the oldest. It provides patient care hospitalization services.")

        st.write("**:black[Department 2]**")
        st.write(
            "The second department is the newest. It comprises three compartments: administration, pharmacy, and laboratory.")
    #image = Image.open('centre.jpg')
    #st.image(image, width= 400)

if rad == "Aim of study":

    st.write("**:green[Definition of Sickle Cell Disease]**")

    st.write(
        "In some cases, chronic anemia can progress to potentially life-threatening acute anemia, especially in children under 7 years old. Red blood cells accumulate in the spleen, where they are rapidly destroyed. This condition is known as 'splenic sequestration' and is specific to children with sickle cell disease. The spleen suddenly enlarges (splenomegaly) and becomes painful. The affected person exhibits significant paleness. Without prompt blood transfusion, the oxygenation of the brain and organs in general can become insufficient, leading to death.")
    st.write("**:green[Aim of the study]**")

    st.write(
        "This application aims to predict whether a sickle cell patient is likely to develop splenic sequestration using artificial intelligence.")

if rad == "My Data":
    if st.checkbox("Show Table"):
        column_names = df.columns.tolist()

        # Assign original column names to DataFrame
        df.columns = column_names

        # Display the modified data frame
        st.dataframe(df)

    graph = st.selectbox("What kind of Graph ? ", ["Non-Interactive", "Interactive"])
    if graph == "Non-Interactive":
        mypal = ["#3CB371", "#F6F5F4"]  # Example palette with medium green and light yellow
        num_feats = ['agediagnosticmois', 'tailleratebasecm', 'hbbasale', 'dosemgkgj', 'age1', 'délaiséquestration']

        L = len(num_feats)
        ncol = 2
        nrow = int(np.ceil(L / ncol))

        # Create a Streamlit figure
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 14), facecolor='#F6F5F4')
        axes = axes.ravel()  # Flatten the axes array

        # Set the title
        st.title('Distribution of Numerical Features')

        # Iterate through each feature
        for i, col in enumerate(num_feats):
            ax = axes[i]  # Select the current axis

            # Plot the KDE plot
            sns.kdeplot(data=df, x=col, hue="séquestration", multiple="stack", palette=mypal, ax=ax)
            ax.set_xlabel(col, fontsize=20)
            ax.set_ylabel("density", fontsize=20)
            sns.despine(right=True)
            sns.despine(offset=0, trim=False)

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Display the figure in Streamlit
        st.pyplot(fig)

    if graph == "Interactive":
        import plotly.graph_objects as go
        import streamlit as st

        num_feats = ['agediagnosticmois', 'tailleratebasecm', 'hbbasale', 'dosemgkgj', 'age1', 'délaiséquestration']

        # Set the title
        st.title('Distribution of Numerical Features')

        # Create a Plotly figure
        fig = go.Figure()

        # Iterate through each feature
        for col in num_feats:
            # Add a box plot trace
            fig.add_trace(go.Box(y=df[col], name=col))

        # Update layout
        fig.update_layout(
            yaxis_title='Feature',
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig)

if rad == "Prediction":

    import pickle
    import streamlit as st
    import pandas as pd

    # Load the trained model
    model = pickle.load(open('C:/Users/Ameni/PycharmProjects/drepano1/model.pkl3', 'rb'))

    st.write('Enter the values for each feature to get the prediction.')

    # Create input fields for each feature
    hydroxyuree = st.selectbox('hydroxyuree', [1, 0, "no input"], index=2)
    transfusionsanguinesipmle = st.number_input('transfusionsanguinesipmle', step=1.0, value=0.0)
    etp = st.selectbox('etp', ['no input', 'non', 'oui'], index=0)
    episodessequestration = st.selectbox('episodessequestration', ['no input', 'autres épisodes', 'premier episode'],
                                         index=0)
    ratebasesup2 = st.selectbox('ratebasesup2', [1, 0, "no input"], index=2)
    rateabsenteaudiagnostic = st.selectbox('rateabsenteaudiagnostic', [1, 0, "no input"], index=2)
    CVOaprèsspc = st.selectbox('CVOaprèsspc', ['no input', 'non', 'oui'], index=0)
    CVOavantsplenectomie = st.selectbox('CVOavantsplenectomie', [1, 0, "no input"], index=2)
    dhtr = st.selectbox('dhtr', ['no input', 'NON', 'OUI', 'autre'], index=0)
    deficitengpd = st.selectbox('deficitengpd', ['no input', 'autre', 'non', 'oui'], index=0)

    # Map categorical feature values to numerical representations
    etp_mapping = {'non': 0, 'oui': 1, 'no input': 0}
    episodessequestration_mapping = {'autres épisodes': 0, 'premier episode': 1, 'no input': 0}
    CVOaprèsspc_mapping = {'non': 0, 'oui': 1, 'no input': 1}
    dhtr_mapping = {'NON': 1, 'OUI': 0, 'autre': 0, 'no input': 1}
    deficitengpd_mapping = {'autre': 0, 'non': 1, 'oui': 0, 'no input': 1}

    # Replace categorical feature values with numerical representations
    etp = etp_mapping[etp]
    episodessequestration = episodessequestration_mapping[episodessequestration]
    CVOaprèsspc = CVOaprèsspc_mapping[CVOaprèsspc]
    dhtr = dhtr_mapping[dhtr]
    deficitengpd = deficitengpd_mapping[deficitengpd]

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'hydroxyuree': [1 if hydroxyuree == 'no input' else hydroxyuree],
        'transfusionsanguinesipmle': [transfusionsanguinesipmle],
        'etp': [etp],
        'episodessequestration': [episodessequestration],
        'ratebasesup2': [1 if ratebasesup2 == 'no input' else ratebasesup2],
        'rateabsenteaudiagnostic': [1 if rateabsenteaudiagnostic == 'no input' else rateabsenteaudiagnostic],
        'CVOaprèsspc': [CVOaprèsspc],
        'CVOavantsplenectomie': [1 if CVOavantsplenectomie == 'no input' else CVOavantsplenectomie],
        'dhtr_NON': [dhtr],
        'deficitengpd_autre': [deficitengpd]
    })

    # Create a submit button
    submit_button = st.button('Submit')

    # Calculate and display the prediction when the button is clicked
    if submit_button:
        # Make prediction
        prediction = model.predict(input_data)

        # Convert numerical prediction to categorical value
        prediction_label = 'No Crisis' if prediction[0] == 0 else 'Crisis'

        # Display the prediction
        st.subheader('Prediction')
        st.write(prediction_label)

















