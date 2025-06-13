
import zipfile
import os

import pandas as pd

from dotenv import load_dotenv
from unidecode import unidecode
from sqlalchemy import create_engine

from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

from langchain_community.utilities import SQLDatabase

#carrego chave secreta para acesso à API
load_dotenv()

def criar_agente_especializado(modo = 'csv_agent'):

    #Criaremos um agente csv/sql, para interpretar os dados como uma tabela diretamente (no caso, no sql)
    if modo == 'csv_agent':
        
        #carrego os dados nos dataframes sem extraí-los propriamente
        with zipfile.ZipFile("202401_NFs.zip", "r") as zip_ref:
            df_NF = pd.DataFrame()
            for file in zip_ref.namelist():
                with zip_ref.open(file) as f:
                    if 'Cabe' in file:
                        cabecalhos = pd.read_csv(f)
                    elif 'Iten' in file:
                        itens = pd.read_csv(f)
        #garanto merge
        cabecalhos['CHAVE DE ACESSO'] = cabecalhos['CHAVE DE ACESSO'].str.strip()
        itens['CHAVE DE ACESSO'] = itens['CHAVE DE ACESSO'].str.strip()

        #colunas diferentes
        dif_cols = list(set(cabecalhos.columns).difference(set(itens.columns)))
        df_NF = itens.merge(cabecalhos[['CHAVE DE ACESSO']+dif_cols], on=['CHAVE DE ACESSO'], how='inner')

        #ajeitando colunas
        new_cols = [unidecode(col.lower().replace(" ","_")) for col in df_NF.columns]
        df_NF.columns = new_cols
        #ajeitando tipo de dados
        df_NF['data_emissao'] = pd.to_datetime(df_NF['data_emissao'])
        df_NF['data/hora_evento_mais_recente'] = pd.to_datetime(df_NF['data/hora_evento_mais_recente'])
        print('Dados prontos')

        #vamos criar a base em sql
        engine = create_engine("sqlite:///NF.db")
        if not os.path.exists(os.getcwd()+'\\NF.db'):
            df_NF.to_sql("NF", engine, index=False)

        db = SQLDatabase(engine=engine)

    #Instanciando o agente de IA
    llm = ChatOpenAI(
        model_name = "gpt-4.1-mini",
        temperature=0
    )
    #criando agente especializado
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    #permito que o usuario faça perguntas à llm, que já possui o contexto
    prompt_usuario = input()
    while prompt_usuario != 'sair':
        response = agent_executor.invoke({'input': prompt_usuario})
        print(response['text'])
        prompt_usuario = input()

    return

if __name__ == '__main__':

    criar_agente_especializado()