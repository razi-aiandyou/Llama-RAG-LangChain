import os
import pandas as pd
from langchain_community.document_loaders import UnstructuredPDFLoader, CSVLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase
import duckdb
from .file_manager import UPLOAD_FOLDER
from pi_heif import register_heif_opener
from sqlalchemy import create_engine

register_heif_opener()

def load_csv_to_sql(csv_file_path, engine):
    df = pd.read_csv(csv_file_path, encoding="utf-8", on_bad_lines="skip", low_memory=False, sep=';')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    table_name = os.path.splitext(os.path.basename(csv_file_path))[0].replace(' ', '_').lower()
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    return table_name

def create_sql_database_from_csv(csv_file_path):
    try:
        #engine = create_engine('duckdb:///:memory:')

        engine = create_engine("sqlite:///uploaded_files.db")
        table_name = load_csv_to_sql(csv_file_path, engine)
        db = SQLDatabase(engine)

        return db, table_name
    except Exception as e:
        print(f"Error processing CSV {csv_file_path}: {str(e)}")
        return None, None

def load_and_process_document(file_path):
    if file_path.endswith('.csv'):
        return create_sql_database_from_csv(file_path)
    elif file_path.endswith('.pdf'):
        loader = UnstructuredPDFLoader(file_path)
    elif file_path.endswith('.txt') or file_path.endswith('.md'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.json'):
        loader = JSONLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits, None

def process_files(embedding_model):
    vectorstore = FAISS.from_texts([""], embedding=embedding_model)
    sql_databases = {}

    for file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file)
        try:
            data, table_name = load_and_process_document(file_path)
            if table_name and data:
                sql_databases[table_name] = data
            elif data:
                vectorstore.add_documents(data)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    return vectorstore, sql_databases
