#!/usr/bin/env python
# coding: utf-8

# In[5]:
import argparse
import torch
from transformers import ViTModel, ViTImageProcessor, AutoModel, ViTImageProcessor
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd
import numpy




def load_models(modelo):

    if modelo == 'ViT_huge':
        vit_huge_model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
        vit_huge_feature_extractor = ViTImageProcessor.from_pretrained('google/vit-huge-patch14-224-in21k')
        vit_huge_model.eval()

        return vit_huge_model, vit_huge_feature_extractor

    if modelo == 'ViT_large':
        vit_large_model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
        vit_large_feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
        vit_large_model.eval()

        return vit_large_model, vit_large_feature_extractor
    
    if modelo == 'ViT_base':
        vit_base_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        vit_base_feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        vit_base_model.eval()
    
        return vit_base_model, vit_base_feature_extractor

    if modelo == 'ViT_small':
        vit_small_model = AutoModel.from_pretrained('WinKawaks/vit-small-patch16-224')
        vit_small_feature_extractor = ViTImageProcessor.from_pretrained('WinKawaks/vit-small-patch16-224')
        vit_small_model.eval()

        return vit_small_model, vit_small_feature_extractor

    return 


# In[33]:


def feature_extraction(image_path, modelo, feature_extractor):
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = modelo(**inputs)
    
    last_hidden_state = outputs.last_hidden_state

    features = last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return features


# In[40]:


def features_to_df(folder_path, modelo, feature_extractor):
    data = []
    
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
    
        if os.path.isdir(subfolder_path):
            for image_file in tqdm(os.listdir(subfolder_path)):
                image_path = os.path.join(subfolder_path, image_file)
                
                if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
                    features = feature_extraction(image_path, modelo, feature_extractor)
                        
                    data.append([image_path, *features])

    columns = ['image_path'] + [f'feature_{i}' for i in range(len(features))]
    df = pd.DataFrame(data, columns=columns)
    return df

    


# In[45]:


def save_dataframe_to_csv(df, caminho, nome_arquivo, modelo):
    caminho_salvar = os.path.join(caminho, nome_arquivo)
    df.to_csv(caminho_salvar+'.csv', index=False)
    print(f"Arquivos do {modelo} salvo em: {caminho}")



def main():
    

    parser = argparse.ArgumentParser()

    parser.add_argument('--modelo', type=str , help= r'Insira um dos modelos, com a sintaxe ViT_{small, base, large, huge}')
    parser.add_argument('--origem', type=str , help= 'Insira a pasta de origem dos arquivos tendo a estrutura, Diretório -> Subdiretórios -> Imagens' )
    parser.add_argument('--armazenar', type=str , help= 'Insira o local no qual o arquivo deve ser salvo')
    parser.add_argument('--nome', type=str , help= 'Insira o nome do arquivo a ser salvo')

    args = parser.parse_args()
    modelo = args.modelo
    caminho_absoluto = args.origem
    caminho_absoluto_salvar = args.armazenar
    nome_arquivo = args.nome

    model, fe = load_models(modelo)
    df = features_to_df(caminho_absoluto, model, fe)
    save_dataframe_to_csv(df, caminho_absoluto_salvar, nome_arquivo, modelo)

if __name__ == '__main__':
    main()



