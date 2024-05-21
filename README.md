## [MediAssist🩺: Your AI-powered Health Assistant](https://medi-assist.streamlit.app/)
**Imagine experiencing symptoms and wanting to understand what might be causing them.** MediAssist is a user-friendly web application designed to be your first line of defense. It leverages the power of machine learning to analyze your symptoms and provide insights into potential health concerns. 

**Here's what MediAssist can do for you:**

* **Predict diseases:** Based on the symptoms you enter, MediAssist utilizes machine learning models to predict the most likely disease you might be facing.
* **Gain clarity:** Receive a detailed description of the predicted disease, helping you understand its characteristics.
* **Take precautions:** MediAssist suggests essential precautions you can take to manage the predicted condition until you consult a doctor. 
* **Find relevant doctors:** If you choose to, the application can search for doctors specializing in the predicted disease, aiding you in seeking professional medical advice.

**Getting Started with MediAssist**

**Prerequisites:**

* A computer with an internet connection
* A web browser (Chrome, Firefox, etc.)

**Step-by-Step Guide:**

1. **Clone the Repository:** 
   - If you're unfamiliar with code repositories, visit platforms like GitHub ([https://github.com/](https://github.com/)) to learn how to clone repositories. 
   - Once you have the basics, clone the MediAssist repository to your local machine.

2. **Install Required Libraries:**
   - Open a terminal or command prompt window.
   - Navigate to the directory containing the cloned MediAssist files (model.py and app.py).
   - Type the following command and press Enter to install the necessary libraries:
     ```bash
     pip install -r requirements.txt
     ```

3. **Prepare the Dataset:**
   - MediAssist relies on several data files to function effectively. Make sure you have the following CSV files in the same directory as your code files:
      * `doctor.csv`: Contains information about doctors (name, specialization, etc.)
      * `symptom_Description.csv`: Provides descriptions for various diseases.
      * `symptom_precaution.csv`: Lists recommended precautions for different diseases.
      * `training_data.csv`: The primary dataset used to train the machine learning models. It contains features corresponding to symptoms and a target variable indicating the disease.

4. **Run the Web Application:**
   - In your terminal or command prompt, navigate back to the directory containing the code files.
   - Type the following command and press Enter to launch the MediAssist web application:
     ```bash
     streamlit run app.py
     ```
   - Your web browser should automatically open a new window displaying the MediAssist interface.

**Using MediAssist:**

The MediAssist interface is straightforward and user-friendly. Here's how to interact with it:

1. **Select your symptoms:** On the main screen, you'll find a section labeled "Enter your symptoms." This section provides a multi-select list containing various symptoms. Choose the symptoms you're experiencing by clicking on their checkboxes.

2. **Get your results:** Once you've selected your symptoms, click the "Predict" button. MediAssist will analyze your selections and display the predicted disease, its description, and recommended precautions.

3. **Find relevant doctors (Optional):** If you'd like to explore doctor options, click the "Yes" button next to the "Do you want to consult a doctor?" question. MediAssist will search for doctors specializing in the predicted disease and display their information.

**Important Note:**

MediAssist is designed to be a helpful tool, but it should not replace professional medical advice. If you're experiencing concerning symptoms, always consult a qualified healthcare provider for an accurate diagnosis and treatment plan.

**We hope MediAssist empowers you to take a proactive approach to your health!**
