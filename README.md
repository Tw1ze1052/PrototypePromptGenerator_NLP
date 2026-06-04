# Procesador de Lenguaje Natural Text-to-Image

Este proyecto consiste en un procesador de lenguaje natural desarrollado en Python. Su objetivo es analizar un texto ingresado por el usuario y mostrar paso a paso los resultados de los análisis léxico, sintáctico y semántico.

Además, el programa genera un prompt visual a partir del análisis semántico. Este prompt puede utilizarse como base para un sistema de generación de imágenes tipo text-to-image. En esta versión no se conecta a una API externa de imágenes, solo se simula el resultado guardando el prompt generado en un archivo JSON.

## Objetivo del proyecto

Desarrollar una herramienta sencilla que permita visualizar cómo una oración o fragmento de texto puede ser procesado mediante técnicas básicas de NLP para identificar palabras, estructuras gramaticales, significado y elementos visuales importantes.

## Tecnologías utilizadas

- Python
- spaCy
- Streamlit
- JSON

## Funcionalidades principales

- Entrada de texto en español o inglés.
- Preprocesamiento del texto.
- Separación del texto en oraciones.
- Análisis léxico:
  - tokens
  - lemas
  - categorías gramaticales
  - dependencias
- Análisis sintáctico:
  - sujetos
  - verbos principales
  - objetos
  - modificadores
  - frases nominales
  - frases preposicionales
- Análisis semántico:
  - personajes o entidades
  - acciones
  - objetos
  - lugares o contextos
  - atributos
  - atmósfera
- Generación de un prompt visual.
- Guardado de resultados en un archivo JSON.
- Interfaz web desarrollada con Streamlit.

## Instalación

Primero se deben instalar las librerías necesarias:

pip install spacy streamlit

Después se deben descargar los modelos de lenguaje de spaCy:

python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm

## Ejecución del programa

Para ejecutar la interfaz se utiliza el siguiente comando en PowerShell:

python -m streamlit run nlp_text_to_image.py

Se abrirá un hipervínculo de acceso local para comprobar su funcionamiento
