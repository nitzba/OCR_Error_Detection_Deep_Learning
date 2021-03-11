# OCR Error Detection 📝

### About the project 📖

---

This is a final project for Digital-Humanities Course taken in Ben Gurion University 2020-2021.<br>
The project attempt to make the process of digitization of laws and legislation in Israel much easier.<br><br>
The project consists of detecting errors in documents that been through the process of OCR.<br>
We used techniques of machine and deep learning in order to detect the errors.<br><br>

The full process and explanations can be found in the following jupyter notebook:<br>
**_OCR-Error_detection_notebook.ipynb_**.<br><br>

For you convenience we have provided a script that uses the model we trained in order to detect errors in a folder that contains **docx** files.
So you can use this script if you are just interested in detecting errors in documents.<br>
The script can be found in the following file: **_ocr_predict_errors.py_**.<br><br>

A summarized report can be found in the following <a href="https://github.com/nitzba/OCR_Error_Detection_Deep_Learning/blob/main/ocr_error_detection_report.pdf"> link </a>.

### Setup ⚙️

---

- Make sure you have python 3.6 or above on your computer.<br>
  We recommend using **Anaconda** that can be downloaded in the following link:<br>
  https://www.anaconda.com/products/individual

- Clone this repository or download and extract the zip of this repository (make sure you have all the files). ***Repository size:** 669 MB - zipped, 1.03 GB - unzipped*

- Open up terminal in the project directory.<br>In order to setup the relevant libraries for the project use the give **_requirements_script.txt_** file by running:<br>

  ```
  pip install -r requirements_script.txt
  ```

### How to run 🏃‍♂️

---

We will explain how to run the two main files we've provided:

1. ocr_predict_errors.py
2. OCR-Error_detection_notebook.ipynb

#### Python script - ocr_predict_errors.py

---

- Open up a terminal in the project directory.
- Create a directory with the **DOCX files** you wish to predict errors upon and name the directory.<br>
  **Note**: make sure that the files are in **DOCX** file format. If you wish to convert some custom files to this format
  use the following link:<br>
  https://www.zamzar.com/convert/doc-to-docx/<br>
  For you convenience, we provided all the original **DOC** files in **DOCX** format in the project.
- Run the following command in the terminal:<br>
  ```
  python ocr_predict_errors.py [path_of_your_docx_folder]
  ```
  **Note**: This process may take a while depending on the amount of files you wish to predict errors upon.<br><br>
  The output will be given in the project directory in the following path: **./text_files/[running_timestamp]** (where [running_timestamp] is the time and date you run the script).<br><br>
  Alternatively, you can append a custom name to the output folder by providing the optional argument as such:
  ```
  python ocr_predict_errors.py [path_of_your_docx_folder] -sn [your_custom_name]
  ```

#### Jupyter Notebook - OCR-Error_detection_notebook.ipynb

---

- In order to run this notebook, make sure you installed jupyter (we recommend using Anaconda).
- Run the following command in the terminal:
  ```
  jupyter notebook
  ```
  and navigate to the project directory using the opened browser.
- Open the notebook in the browser.
- Run the notebook step by step (top to bottom) and view the outputs.<br>
  **Note**: The process of running the notebook may take a long time, we already did all the steps required. So, we recommend viewing the outputs we provided.

### Project output 📤

---

As mentioned, the output will be given in **txt file format** and will be in the directory **text_files** we mentioned above.
<br><br>
The output is constructed from of our **deep learning** model prediction on each docx file.<br>
Each word that our model predicted as an error will be marked with a special tag.<br>
For example:

```
01 ספר החוקים 207 כ״ו באב <e>התשי״ו<e> 03 . 08 . 1956
* נתקבל בכנסת ביום י״ז באב תשט״ז ( 25 ביולי 1956 ) ; הצעת החוק ודברי הסבר נתפרסמו
<e>בה״ח<e> 278 , תשט״ז , עמ׳ 163 .
1 ס״ח <e>ת190<e> , תשט״ו עמ׳ 242 .
לוי אשכול
שר האוצר
דוד בן - גוריון
ראש הממשלה
יצחק בן - צבי
נשיא המדינה
תיקון החוק
1 . בסעיף 1 לחוק בנק ישראל ( הוראת שעה ) ׳ תשט״ו - 1955 1 , במקום ״ <e>כ<e> ״ ו באב תשט״ז
0 : באוגוסט 1956 ) ״ יבוא ״ ב׳ באדר א׳ תשי״ז ( 3 בפברואר 1957 ) ״ .
מספר 59
חוק בנק ישראל ( הוראת שעה ) ( תיקון מס׳ 2 ) , תשט״ז - 1956 <e>*<e>
```
