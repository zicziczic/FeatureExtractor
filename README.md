python = 3.10.12

python -m venv env

source env/bin/activate

Se usar jupyter (ipython kernel install --user --name=env)

instalar do link "https://pytorch.org/get-started/locally/#linux-installation"
(Verificar versões)

pip install pandas
pip install transformers

para rodar 
python FeatureExtractor.py
	--modelo Insira um dos modelos, com a sintaxe ViT_{small, base, large, huge}
	--origem Insira a pasta de origem dos arquivos tendo a estrutura
				    Diretório -> Subdiretórios -> Imagens
	--armazenar Insira o local no qual o arquivo deve ser salvo
	--nome Insira o nome do arquivo a ser salvo
