import argparse
import os
import pickle
import re
from datetime import datetime

import docx2txt
import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout, Embedding, TimeDistributed
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sentence_length = 50
batch_size = 200


def replace_quotes(data):
    """
    Replacing the char " with ״ in a text.
    """
    data = re.sub('"', '״', data)
    return data


def remove_extra_newlines(data, akn=True):
    """
    Removes more than one newline in a text.

    Args:
        data (String): text we wish to remove newlines from.
        akn (bool): True if the @data came from akn xml file.

    Returns: the text without extra newlines.
    """
    if akn:
        data = data[1:]
    data = re.sub(r'\n\s*\n', '\n', data)
    return data


def replace_dots(data):
    """
    add space between colons in @data
    """
    data = re.sub(':', ' : ', data)
    return data


def replace_dot_coma(data):
    """
    add space between semi-colons in @data
    """
    data = re.sub(';', ' ; ', data)
    return data


def replace_coma(data):
    data = re.sub('[,]', ' , ', data)
    return data


def replace_dot(data):
    """
    replacing dot surrounded square brackets with dot surrounded by spaces
    """
    data = re.sub("[.]", " . ", data)
    return data


def replace_extra_whitespaces_into_one(data):
    data = re.sub(' +', ' ', data)
    return data


def make_space_quotes(data, words_with_quotes):
    """
    adding space to the start and the end of each word containing quotes which are not in @words_with_quotes.
    """
    sentences = data.split('\n')
    new_sentences = []
    for sen in sentences:
        words = sen.split()
        new_words = []
        for word in words:
            splitted_by_quote = word.split("״")
            bounds = [0, len(word) - 1]
            if word not in words_with_quotes and len(splitted_by_quote) > 0 and word.find("״") in bounds:
                new_word = " ״ ".join(splitted_by_quote).strip()
                new_words.append(new_word)
            else:
                new_words.append(word)
        new_sentences.append(" ".join(new_words))
    return "\n".join(new_sentences)


def replace_more_special_characters(data):
    data = re.sub("[?]", " ? ", data)
    data = re.sub("[!]", " ! ", data)
    data = re.sub("[%]", " % ", data)
    data = re.sub("[*]", " * ", data)
    data = re.sub("[/]", " / ", data)

    data = re.sub("–", " – ", data)
    data = re.sub("<", " < ", data)
    data = re.sub(">", " > ", data)

    data = re.sub("־", "-", data)
    data = re.sub("-", " - ", data)
    return data


def replace_quotes_with_quotes_and_spaces(data):
    """
    Replacing all the quotes in @data with ' “ '
    """
    data_ = data.split('\n')
    for j, line in enumerate(data_):
        words = line.split(' ')
        for i, word in enumerate(words):
            quote_left = word.find('“')
            quote_right = word.find('”')
            word = list(word)
            if quote_left != -1:
                word[quote_left] = ' “ '
            if quote_right != -1:
                word[quote_right] = ' “ '
            word = ''.join(word)
            words[i] = word
        line = ' '.join(words)
        data_[j] = line
    data = '\n'.join(data_)
    return data


def replace_brackets(data):
    data = re.sub("[(]", " ( ", data)
    data = re.sub("[)]", " ) ", data)
    data = re.sub("[[]", " [ ", data)
    data = re.sub("[]]", " ] ", data)
    return data


def replace_single_quote(data):
    """
    Replacing the char ' with ׳ in a text.
    """
    data = re.sub("'", '׳', data)
    return data


def get_words_with_quotes():
    with open('./wq/words_with_quotes.pickle', 'rb') as handle:
        wq = pickle.load(handle)
        return wq


def preprocess_text(df, text_column_name="law_akn_text"):
    """
    Performs preprocessing on all the text in a dataframe

    Args:
        df (DataFrame): the dataframe we wish to perform preprocessing on.
        text_column_name: column name in @df we wish to perform the preprocessing on.

    Returns: The original dataframe after preprocessing.
    """
    words_with_quotes = get_words_with_quotes()

    # removing extra newlines from all the text
    df[text_column_name] = df[text_column_name].apply(
        lambda x: remove_extra_newlines(x))
    # replacing special double quotes with regular in all the text
    df[text_column_name] = df[text_column_name].apply(
        lambda x: replace_quotes(x))
    # replacing special quote with regular in all the text
    df[text_column_name] = df[text_column_name].apply(
        lambda x: replace_single_quote(x))
    # padding colon with spaces
    df[text_column_name] = df[text_column_name].apply(
        lambda x: replace_dots(x))
    # padding semi-colon with spaces
    df[text_column_name] = df[text_column_name].apply(
        lambda x: replace_dot_coma(x))
    # padding dot with spaces
    df[text_column_name] = df[text_column_name].apply(lambda x: replace_dot(x))
    # padding brackets with spaces
    df[text_column_name] = df[text_column_name].apply(
        lambda x: replace_brackets(x))
    # padding commas with spaces
    df[text_column_name] = df[text_column_name].apply(
        lambda x: replace_coma(x))
    # padding brackets with spaces
    df[text_column_name] = df[text_column_name].apply(
        lambda x: replace_more_special_characters(x))
    # replacing all the quotes with a generic one
    df[text_column_name] = df[text_column_name].apply(
        lambda x: replace_quotes_with_quotes_and_spaces(x))
    # padding words containing quotes with spaces
    df[text_column_name] = df[text_column_name].apply(
        lambda x: make_space_quotes(x, words_with_quotes))
    # removing extra whitespace
    df[text_column_name] = df[text_column_name].apply(
        lambda x: replace_extra_whitespaces_into_one(x))
    return df


def get_law_docx_text(law_docx_path):
    """
    Returns the text in a DOCX file.
    """
    text = docx2txt.process(law_docx_path)
    return text


def create_docx_df(docx_path):
    """
    Args:
        docx_path - path the the dir where all the docx file located

    Return:
        docx_df - This is data frame contains all the docx files texts and their laws id.
    """
    docx_texts = []
    file_names = os.listdir(docx_path)

    for file_name in tqdm(file_names, desc="Reading DOCX"):
        law_id = file_name.split(".")[0]
        try:
            law_docx_path = f'{docx_path}/{file_name}'
            law_docx_text = get_law_docx_text(law_docx_path)

            docx_texts.append({
                "law_id": law_id,
                "law_docx_text": law_docx_text,
            })
        except Exception as e:
            pass

    law_docx_df = pd.DataFrame(docx_texts, columns=["law_id", "law_docx_text"])
    return law_docx_df


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer


def create_batch_size_matrices(docx_law_to_lines, tokenizer):
    """
    Args: 
        docx_law_to_lines - The law text file splitted into lines
        tokenizer - Use to convert the text file words to tokens (with the same dictionary)

    Return: 
        X - Array of matrices which each matrix is in the size of (batch_size, sentence_length)
    """
    text_len = len(docx_law_to_lines)
    index = 0

    X = []
    while True:
        if text_len > batch_size:
            mat = create_batch_matrix_X(
                docx_law_to_lines[batch_size * index:batch_size * (index + 1)], tokenizer)
            X.append(mat)
            text_len -= batch_size
            index += 1
        else:
            mat = create_batch_matrix_X(
                docx_law_to_lines[batch_size * index:len(docx_law_to_lines)], tokenizer)
            X.append(mat)
            break

    return np.array(X)


def create_batch_matrix_X(text, tokenizer):
    """
    params: 
         text - the text of the file
         tokenizer - Use to tokenize each sentence.
    return: 
        return matrix X of (batch_size, sentence_length) with padding to the sentences.
    """
    mat_x = np.zeros((batch_size, sentence_length))

    for i, line in enumerate(text):
        line = re.sub('\t\n', '', line)
        line_seq = tokenizer.texts_to_sequences([line])
        line_seq = pad_sequences(
            line_seq, maxlen=sentence_length, padding='post')
        mat_x[i] = np.array(line_seq[0], dtype=int)

    return mat_x


def prepare_data_to_predict(docx_df, tokenizer):
    """
    Args:
        docx_df - docx data frame, contains all the docx texts.

    Return:
        docx_files_num_of_batches - list of all the docx text file number of batchs to each text.
        Each text can be splitted into more than one batch size.
        docx_batch_matrices - list of all texts files splitted into batch size matrices.
    """
    docx_files_num_of_batches = []

    docx_text_splited = docx_df["law_docx_text"][0].split('\n')
    docx_batch_matrices = create_batch_size_matrices(
        docx_text_splited, tokenizer)
    docx_files_num_of_batches.append(docx_batch_matrices.shape[0])

    docx_df = docx_df[1:]
    for _, row in tqdm(docx_df.iterrows(), desc="Preparing DOCX to Predict"):
        docx_text = row["law_docx_text"]
        docx_text_splited = docx_text.split('\n')

        docx_batch_matrices_curr = create_batch_size_matrices(
            docx_text_splited, tokenizer)

        docx_files_num_of_batches.append(docx_batch_matrices_curr.shape[0])

        docx_batch_matrices = np.vstack(
            (docx_batch_matrices, docx_batch_matrices_curr))

    return docx_files_num_of_batches, docx_batch_matrices


def create_df_predict_test_docx(docx_df, predicted_docx, docx_files_batches_vec):
    """
    Args:
        docx_df - docx data frame, contains all the docx texts.
        predicted_docx - model output on the docx text files.
        docx_files_batches_vec - list of number of batch sizes for each docx file. 

    Return:
        docx_and_matrices_df - data frame that contains the law id, the original text of the law and the predicted
        matrix from the model.
    """
    docx_and_matrices = []

    for i, row in tqdm(docx_df.iterrows(), desc="Creating DOCX df for predict"):
        id_docx = row["law_id"]
        text_docx = row["law_docx_text"]
        predicted_docx_matrix = predicted_docx[:
                                               docx_files_batches_vec[i] * batch_size]

        docx_and_matrices.append({
            "law_id": id_docx,
            "damaged_text": text_docx,
            "damaged_words_sen": predicted_docx_matrix,
        })

        predicted_docx = predicted_docx[docx_files_batches_vec[i]
                                        * batch_size:]

    docx_and_matrices_df = pd.DataFrame(docx_and_matrices, columns=[
                                        "law_id", "damaged_text", "damaged_words_sen"])
    return docx_and_matrices_df


def mark_word(word):
    return f"<e>{word}<e>"


def get_marked_line(words, damaged_vec):
    marked_words = []

    for i, word in enumerate(words):
        if i < sentence_length:
            marked_word = word if damaged_vec[i] == 0 else mark_word(word)
            marked_words.append(marked_word)
        else:
            marked_words.append(word)
    return marked_words


def mark_text_with_error_tag(law_text, law_mat):
    """
    Mark a text file with an <e> tag.

    Args:
        law_text (String): text that suspected to be containing errors.
        law_mat (numpy.array): a matrix that indicates where the errors are in the text.

    Returns:
        The text marked with <e> tag where @law_mat suggested.
    """
    law_lines = law_text.split("\n")
    marked_lines = []
    for i, line in enumerate(law_lines):
        curr_damaged_vec = law_mat[i]
        curr_words = line.split()
        marked_line = get_marked_line(curr_words, curr_damaged_vec)
        marked_line = " ".join(marked_line)
        marked_lines.append(marked_line)
    return "\n".join(marked_lines)


def mark_texts_with_error_tag(damaged_texts_df):
    """
    Marks all the words a model predicted as damaged with an <error> tag.

    damaged_texts_df (DataFrame): dataframe contaning the docx text 
    and the places where the model predicted the words are damaged
    """
    marked_texts = []

    for _, row in tqdm(damaged_texts_df.iterrows(), desc="Marking Words"):
        law_id = row["law_id"]
        law_damaged = row["damaged_text"]
        law_damaged_mat = row["damaged_words_sen"]
        law_marked = mark_text_with_error_tag(law_damaged, law_damaged_mat)

        marked_texts.append({
            "law_id": law_id,
            "text": law_marked,
        })
    return pd.DataFrame(marked_texts, columns=["law_id", "text"])


def save_text_file(text, path):
    """
    Saves a text file in path

    Args:
        text (String): text to be saved.
        path (String): where to save the text file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def save_text_files(texts_df, name=""):
    """
    Saving a dataframe containing text files to a folder.

    Args:
        texts_df (DataFrame): dataframe containing text files.
    """
    now_time = datetime.now()
    now_str = now_time.strftime("%Y_%m_%d_%H_%M")
    base_file_path = f"./text_files/{name}_{now_str}"
    for _, row in tqdm(texts_df.iterrows(), desc="Saving..."):
        law_id = row["law_id"]
        law_text = row["text"]
        file_path = f'{base_file_path}/{law_id}.txt'
        try:
            save_text_file(law_text, file_path)
        except:
            pass


def get_model(vocab_size):
    # hyperparameters for the model
    hidden_size = 100
    units_lstm = 125

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=hidden_size,
                        input_length=sentence_length, trainable=False))
    model.add(LSTM(units=units_lstm, return_sequences=True))
    model.add(LSTM(units=units_lstm, return_sequences=True))
    model.add(LSTM(units=units_lstm, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1, activation="sigmoid")))


def weighted_binary_crossentropy_v1(w1, w2):
    def loss(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        logloss = -(y_true * K.log(y_pred) * w1 +
                    (1 - y_true) * K.log(1 - y_pred) * w2)
        return K.mean(logloss, axis=-1)

    return loss


def ocr_predict_errors_pipline(docx_path, name_to_save=""):
    tokenizer = load_tokenizer("./tokenizer/tokenizer.pickle")
    model = get_model(len(tokenizer.word_index) + 1)
    model = keras.models.load_model("./models/try1_weights.20-0.02713.hdf5",
                                    custom_objects={'loss': weighted_binary_crossentropy_v1(3, 1)})

    docx_df = create_docx_df(docx_path)
    law_docx_df_clean = preprocess_text(
        docx_df, text_column_name="law_docx_text")
    docx_files_batches_vec, docx_batch_matrices = prepare_data_to_predict(
        law_docx_df_clean, tokenizer)
    X_test_docx = docx_batch_matrices.reshape((-1, sentence_length))
    # predicting on the docx files
    preds_docx = model.predict(X_test_docx)
    # mapping back the predictions to the files
    docx_and_matrices_df = create_df_predict_test_docx(
        law_docx_df_clean, preds_docx.round(), docx_files_batches_vec)
    marked_df = mark_texts_with_error_tag(docx_and_matrices_df)
    save_text_files(marked_df, name=name_to_save)


def get_command_line_args():
    """
    Returns the application arguments parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'docx_path', help="path to directory containing docx files")
    parser.add_argument(
        '-sn', '--save_name', help="name of the directory containing the marked text files")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_command_line_args()
    docx_path = args.docx_path
    save_name = args.save_name if args.save_name else ""
    ocr_predict_errors_pipline(docx_path, name_to_save=save_name)
