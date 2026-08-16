import pandas as pd 
import numpy as np 
import random as rd 
import pymupdf
import spacy
import os 
import re 
import shutil
from functools import partial
from toolz import compose,compose_left
import unicodedata
import pprint




class Classe_criador_chunks() :
    def __init__(self,query) :
        self.query = query
        self.desencadear()


    def desencadear(self) :
        self.criar_parametros()
        self.extracao_pdf()
        self.criacao_chunks()
        self.busca_semantica()
        self.busca_lexica()
        self.listagem_contexto()


    def normalizar_texto(self,texto=None):
        # 1. Minúsculas
        texto = texto.lower()

        # 2. Remove URLs
        texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto)

        # 3. Remove e-mails
        texto = re.sub(r'\S+@\S+', ' ', texto)

        # 4. Remove acentos
        texto = unicodedata.normalize("NFD", texto)
        texto = "".join(
            caractere
            for caractere in texto
            if unicodedata.category(caractere) != "Mn"
        )

        # 5. Remove caracteres especiais
        # Mantém letras, números e espaços
        texto = re.sub(r'[^a-z0-9\s]', ' ', texto)

        # 6. Remove espaços duplicados
        texto = re.sub(r'\s+', ' ', texto).strip()

        return texto


    def criar_parametros(self) :
        self.nlp = spacy.load("pt_core_news_sm")


    def extracao_pdf(self) :
        caminho_pdf = r".\pdfs\Guia de empresas juniores.pdf"
        caminho_txt = r".\logs\base_bruta_texto.txt"
        arquivo_pdf = pymupdf.open(caminho_pdf)
        with open(caminho_txt,"w",encoding="utf-8") as arquivo_tt :
            for page in arquivo_pdf :
                text_ref = page.get_text()
                arquivo_tt.write(text_ref)


    def criacao_chunks(self) :
        caminho_txt = r".\logs\base_bruta_texto.txt"
        with open(caminho_txt,"r",encoding="utf-8") as arquivo_tt :
            conteudo = arquivo_tt.read()
        ajustar_texto = partial(self.normalizar_texto,texto=conteudo)
        conteudo_01 = ajustar_texto()
        doc = self.nlp(conteudo_01)
        lista_chunks = []
        lista_frame = []
        x = 1
        for docs in doc :
            lista_chunks.append(docs)
            if len(lista_chunks) >= 258 :
                var_contex = f""""""
                for a in lista_chunks :
                    var_contex = f"""{var_contex} {a}"""
                lista_frame.append({"Chunk":x,"Conteudo":var_contex})
                lista_chunks = []
                x += 1
        frame_final = pd.DataFrame(lista_frame)
        self.frame_chunks = frame_final.copy()
        print(frame_final)


    def busca_semantica(self) :
        normalizar_q = partial(self.normalizar_texto,texto=self.query)
        doc = self.nlp(normalizar_q())
        scores = []
        for i , text_ref in enumerate(self.frame_chunks["Conteudo"]) :
            sm = []
            doc_ref = self.nlp(text_ref)
            for o in doc_ref :
                vetor_ref = doc.similarity(o)
                sm.append(vetor_ref)
            val_max = max(sm)
            scores.append({"Chunk":self.frame_chunks.loc[i,"Chunk"],"Valor":val_max})
        frame_scores = pd.DataFrame(scores)
        frame_rela01 = pd.merge(self.frame_chunks,frame_scores,left_on="Chunk",right_on="Chunk",suffixes=("_x","_y"),how="left")
        frame_rela01 = frame_rela01.sort_values(by="Valor",ascending=False).reset_index(drop=True).head(10)
        self.frame_rela02 = frame_rela01.copy()
        print(frame_rela01)


    def busca_lexica(self) :
        normalizar_q = partial(self.normalizar_texto,texto=self.query)
        lemma_q = {x.lemma_ for x in self.nlp(normalizar_q())}
        lista_final = []
        for i , chunk_ref in enumerate(self.frame_rela02["Conteudo"]) :
            lemma_chun = {x.lemma_ for x in self.nlp(chunk_ref)}
            if lemma_q.intersection(lemma_chun) :
                lista_final.append({"Chunk":self.frame_rela02.loc[i,"Chunk"],"Intencidade":len(lemma_q.intersection(lemma_chun)),"Similaridade":lemma_q.intersection(lemma_chun),"Menssagem":chunk_ref,"Referencia_Vetorial":self.frame_rela02.loc[i,"Valor"]})
        frame_final_cor = pd.DataFrame(lista_final)
        frame_final_cor = frame_final_cor.sort_values(by="Intencidade",ascending=False).reset_index(drop=True)
        self.frame_final_0 = frame_final_cor.copy()
        print(frame_final_cor)



    def listagem_contexto(self) :
        lista_final = []
        for i , chunk_ref in enumerate(self.frame_final_0["Chunk"]) :
            if chunk_ref == 1 :
                lista_final.append({
                    "Chunk":chunk_ref
                })
                lista_final.append({
                    "Chunk":chunk_ref + 1
                })
            else :
                lista_final.append({
                    "Chunk":chunk_ref - 1
                })
                lista_final.append({
                    "Chunk":chunk_ref 
                })
                lista_final.append({
                    "Chunk":chunk_ref + 1
                })
        frame_chunks_ref = pd.DataFrame(lista_final)
        frame_chunks_ref = frame_chunks_ref.drop_duplicates(subset="Chunk").reset_index(drop=True)
        frame_chunks_ref = frame_chunks_ref.sort_values(by="Chunk").reset_index(drop=True)
        frame_chunks_ref1 = pd.merge(frame_chunks_ref,self.frame_chunks,left_on="Chunk",right_on="Chunk",suffixes=("_x","_y"),how="left")
        frame_chunks_refj = frame_chunks_ref1.to_json(orient="records",force_ascii=False,indent=4)
        frame_chunks_ref1.to_json(r".\logs\modelo_js.json",orient="records",force_ascii=False,indent=4)
        pprint.pprint(frame_chunks_refj)


        

Classe_criador_chunks("Por esse objetivo entende-se fomentar o crescimento pessoal e profissional do aluno membro, por meio do oferecimento de serviços de qualidade e a um baixo custo ao mercado")