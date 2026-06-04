import json
import os
import spacy
import streamlit as st


def load_nlp_model(language):
    """
    Carga el modelo NLP según el idioma seleccionado.
    """
    if language == "es":
        return spacy.load("es_core_news_sm")
    elif language == "en":
        return spacy.load("en_core_web_sm")
    else:
        raise ValueError("Idioma no soportado. Usa 'es' para español o 'en' para inglés.")


def get_user_text():
    print("Ingresa el texto a analizar.")
    print("Puede ser una oración simple o un fragmento literario.")
    text = input("\nTexto: ")
    return text


def get_language():
    print("\nSelecciona el idioma del texto:")
    print("1. Español")
    print("2. Inglés")

    option = input("Opción: ").strip()

    if option == "1":
        return "es"
    elif option == "2":
        return "en"
    else:
        print("Opción inválida. Se usará español por defecto.")
        return "es"


def preprocess_text(text):
    """
    Limpia espacios innecesarios.
    """
    text = text.strip()
    text = " ".join(text.split())
    return text


def split_sentences(doc):
    """
    Separa el texto en oraciones.
    """
    sentences = []

    for sent in doc.sents:
        sentences.append(sent.text)

    return sentences


def lexical_analysis(doc):
    """
    Realiza análisis léxico:
    tokens, lemas, categorías gramaticales y entidades.
    """
    lexical_result = []

    for token in doc:
        lexical_result.append({
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dependency": token.dep_,
            "head": token.head.text,
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop
        })

    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return {
        "tokens": lexical_result,
        "entities": entities
    }


def get_full_phrase(token):
    """
    Obtiene la frase completa asociada a un token.
    Por ejemplo:
    token: whale
    frase: the great white whale
    """
    words = list(token.subtree)
    words = sorted(words, key=lambda word: word.i)
    phrase = " ".join([word.text for word in words])
    return phrase


def syntactic_analysis(doc):
    """
    Realiza análisis sintáctico más completo.
    Detecta sujetos, verbos, objetos, frases nominales,
    frases preposicionales y dependencias.
    """
    result = {
        "sentences": [],
        "noun_phrases": [],
        "prepositional_phrases": [],
        "dependencies": []
    }

    for chunk in doc.noun_chunks:
        result["noun_phrases"].append({
            "text": chunk.text,
            "root": chunk.root.text,
            "root_pos": chunk.root.pos_,
            "dependency": chunk.root.dep_
        })

    for token in doc:
        result["dependencies"].append({
            "word": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dependency": token.dep_,
            "head": token.head.text
        })

        if token.pos_ == "ADP":
            phrase = get_full_phrase(token)
            result["prepositional_phrases"].append(phrase)

    for sent in doc.sents:
        sentence_data = {
            "sentence": sent.text,
            "root_verb": None,
            "subjects": [],
            "objects": [],
            "modifiers": [],
            "actions": []
        }

        for token in sent:
            if token.dep_ == "ROOT":
                sentence_data["root_verb"] = token.text

            if token.pos_ in ["VERB", "AUX"]:
                sentence_data["actions"].append({
                    "verb": token.text,
                    "lemma": token.lemma_,
                    "dependency": token.dep_
                })

            if token.dep_ in ["nsubj", "nsubjpass", "nsubj:pass"]:
                sentence_data["subjects"].append(get_full_phrase(token))

            if token.dep_ in ["obj", "dobj", "iobj", "pobj", "attr", "oprd"]:
                sentence_data["objects"].append(get_full_phrase(token))

            if token.dep_ in ["amod", "advmod", "acl", "relcl", "compound"]:
                sentence_data["modifiers"].append({
                    "word": token.text,
                    "modifies": token.head.text,
                    "type": token.dep_
                })

        result["sentences"].append(sentence_data)

    return result


def semantic_analysis(doc, syntactic_result):
    """
    Construye una interpretación semántica útil para generar imagen.
    """
    semantic_result = {
        "characters_or_entities": [],
        "actions": [],
        "objects": [],
        "places_or_contexts": [],
        "attributes": [],
        "named_entities": [],
        "atmosphere_words": []
    }

    for ent in doc.ents:
        semantic_result["named_entities"].append({
            "text": ent.text,
            "label": ent.label_
        })

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "PRON"]:
            phrase = get_full_phrase(token)

            if phrase not in semantic_result["characters_or_entities"]:
                semantic_result["characters_or_entities"].append(phrase)

        if token.pos_ == "VERB":
            semantic_result["actions"].append({
                "verb": token.text,
                "lemma": token.lemma_
            })

        if token.pos_ == "ADJ":
            semantic_result["attributes"].append({
                "attribute": token.text,
                "describes": token.head.text
            })

        if token.dep_ in ["obj", "dobj", "pobj", "attr"]:
            phrase = get_full_phrase(token)
            if phrase not in semantic_result["objects"]:
                semantic_result["objects"].append(phrase)

        if token.dep_ in ["obl", "pobj"] or token.pos_ == "ADP":
            phrase = get_full_phrase(token)
            if phrase not in semantic_result["places_or_contexts"]:
                semantic_result["places_or_contexts"].append(phrase)

    atmosphere_keywords = [
        "dark", "storm", "stormy", "sea", "ocean", "night", "cold",
        "silent", "mysterious", "ancient", "lonely", "vast", "deep",
        "oscuro", "tormenta", "mar", "océano", "noche", "frío",
        "silencioso", "misterioso", "antiguo", "solitario", "profundo"
    ]

    for token in doc:
        if token.lemma_.lower() in atmosphere_keywords or token.text.lower() in atmosphere_keywords:
            semantic_result["atmosphere_words"].append(token.text)

    return semantic_result


def generate_visual_prompt(semantic_result, language):
    """
    Genera un prompt visual más rico a partir del análisis semántico.
    """
    entities = semantic_result.get("characters_or_entities", [])
    actions = semantic_result.get("actions", [])
    objects = semantic_result.get("objects", [])
    places = semantic_result.get("places_or_contexts", [])
    attributes = semantic_result.get("attributes", [])
    atmosphere = semantic_result.get("atmosphere_words", [])
    named_entities = semantic_result.get("named_entities", [])

    main_entities = entities[:3]
    main_actions = [action["lemma"] for action in actions[:3]]
    main_objects = objects[:3]
    main_places = places[:2]
    main_attributes = [item["attribute"] for item in attributes[:5]]
    main_named_entities = [item["text"] for item in named_entities[:3]]

    if language == "en":
        prompt = "A detailed visual scene"
    else:
        prompt = "Una escena visual detallada"

    if main_entities:
        prompt += " featuring " + ", ".join(main_entities) if language == "en" else " con " + ", ".join(main_entities)

    if main_named_entities:
        prompt += ", including " + ", ".join(main_named_entities) if language == "en" else ", incluyendo " + ", ".join(main_named_entities)

    if main_actions:
        prompt += ", performing actions such as " + ", ".join(main_actions) if language == "en" else ", realizando acciones como " + ", ".join(main_actions)

    if main_objects:
        prompt += ", with important objects such as " + ", ".join(main_objects) if language == "en" else ", con objetos importantes como " + ", ".join(main_objects)

    if main_places:
        prompt += ", set in " + ", ".join(main_places) if language == "en" else ", ambientada en " + ", ".join(main_places)

    if main_attributes:
        prompt += ", with visual attributes like " + ", ".join(main_attributes) if language == "en" else ", con atributos visuales como " + ", ".join(main_attributes)

    if atmosphere:
        prompt += ", atmosphere: " + ", ".join(atmosphere) if language == "en" else ", atmósfera: " + ", ".join(atmosphere)

    if language == "en":
        prompt += ", cinematic lighting, dramatic composition, detailed digital illustration"
    else:
        prompt += ", iluminación cinematográfica, composición dramática, ilustración digital detallada"

    return prompt


def generate_image(prompt):
    """
    Simula la generación de imagen.
    No llama a ninguna API externa.
    Guarda el prompt visual en un archivo de texto.
    """

    os.makedirs("outputs", exist_ok=True)

    prompt_path = os.path.join("outputs", "generated_prompt.txt")

    with open(prompt_path, "w", encoding="utf-8") as file:
        file.write(prompt)

    return prompt_path


def save_results(data):
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/analysis_result.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print("\nResultados guardados en outputs/analysis_result.json")


def show_results(data):
    print("\n====================================")
    print("RESULTADOS DEL PROCESADOR NLP")
    print("====================================")

    print("\nTexto original:")
    print(data["original_text"])

    print("\nTexto preprocesado:")
    print(data["clean_text"])

    print("\nIdioma:")
    print(data["language"])

    print("\n--- Oraciones detectadas ---")
    for i, sentence in enumerate(data["sentences"], start=1):
        print(f"{i}. {sentence}")

    print("\n--- Análisis léxico ---")
    for item in data["lexical_analysis"]["tokens"]:
        print(
            f"{item['token']} → "
            f"lema: {item['lemma']} | "
            f"categoría: {item['pos']} | "
            f"dependencia: {item['dependency']} | "
            f"depende de: {item['head']}"
        )

    print("\n--- Entidades nombradas ---")
    if data["lexical_analysis"]["entities"]:
        for ent in data["lexical_analysis"]["entities"]:
            print(f"{ent['text']} → {ent['label']}")
    else:
        print("No se detectaron entidades nombradas.")

    print("\n--- Análisis sintáctico por oración ---")
    for i, sentence_data in enumerate(data["syntactic_analysis"]["sentences"], start=1):
        print(f"\nOración {i}: {sentence_data['sentence']}")
        print(f"Verbo raíz: {sentence_data['root_verb']}")
        print(f"Sujetos: {sentence_data['subjects']}")
        print(f"Objetos: {sentence_data['objects']}")
        print(f"Acciones: {sentence_data['actions']}")
        print(f"Modificadores: {sentence_data['modifiers']}")

    print("\n--- Frases nominales ---")
    for phrase in data["syntactic_analysis"]["noun_phrases"]:
        print(f"{phrase['text']} → núcleo: {phrase['root']} | dependencia: {phrase['dependency']}")

    print("\n--- Frases preposicionales ---")
    for phrase in data["syntactic_analysis"]["prepositional_phrases"]:
        print(phrase)

    print("\n--- Análisis semántico ---")
    semantic = data["semantic_analysis"]

    print("Personajes o entidades:")
    print(semantic["characters_or_entities"])

    print("\nAcciones:")
    print(semantic["actions"])

    print("\nObjetos:")
    print(semantic["objects"])

    print("\nLugares o contextos:")
    print(semantic["places_or_contexts"])

    print("\nAtributos:")
    print(semantic["attributes"])

    print("\nAtmósfera:")
    print(semantic["atmosphere_words"])

    print("\n--- Prompt visual generado ---")
    print(data["visual_prompt"])

    print("\n--- Imagen ---")
    print(data["image_path"])

def get_semantic_descriptions():
    return {
        "characters_or_entities": (
            "Identifica los personajes, seres, objetos principales o conceptos "
            "que aparecen en el texto. Estos elementos suelen convertirse en los "
            "protagonistas visuales de la imagen."
        ),
        "actions": (
            "Detecta los verbos o acciones realizadas dentro del texto. "
            "Estas acciones ayudan a representar movimiento, intención o actividad "
            "en la escena generada."
        ),
        "objects": (
            "Extrae los objetos importantes mencionados en la oración. "
            "Estos elementos pueden acompañar a los personajes o formar parte "
            "central de la composición visual."
        ),
        "places_or_contexts": (
            "Reconoce lugares, escenarios o contextos espaciales donde ocurre "
            "la acción. Sirve para definir el fondo o ambiente de la imagen."
        ),
        "attributes": (
            "Identifica características descriptivas como colores, tamaños, "
            "estados o cualidades. Estos atributos ayudan a enriquecer visualmente "
            "a los personajes, objetos y escenarios."
        ),
        "atmosphere_words": (
            "Detecta palabras relacionadas con el ambiente emocional o visual "
            "del texto, como oscuridad, niebla, tormenta, silencio o misterio. "
            "Sirve para definir el tono general de la imagen."
        )
    }


def main():
    original_text = get_user_text()

    if not original_text.strip():
        print("Error: debes ingresar un texto válido.")
        return

    language = get_language()
    nlp = load_nlp_model(language)

    clean_text = preprocess_text(original_text)
    doc = nlp(clean_text)

    sentences = split_sentences(doc)
    lexical_result = lexical_analysis(doc)
    syntactic_result = syntactic_analysis(doc)
    semantic_result = semantic_analysis(doc, syntactic_result)
    visual_prompt = generate_visual_prompt(semantic_result, language)
    image_path = generate_image(visual_prompt)

    data = {
        "original_text": original_text,
        "clean_text": clean_text,
        "language": language,
        "sentences": sentences,
        "lexical_analysis": lexical_result,
        "syntactic_analysis": syntactic_result,
        "semantic_analysis": semantic_result,
        "visual_prompt": visual_prompt,
        "image_path": image_path
    }

    save_results(data)
    show_results(data)


@st.cache_resource
def load_nlp_model_cached(language):
    """
    Carga el modelo NLP una sola vez para mejorar el rendimiento.
    """
    return load_nlp_model(language)


def run_pipeline_from_interface(original_text, language):
    
    clean_text = preprocess_text(original_text)

    nlp = load_nlp_model_cached(language)
    doc = nlp(clean_text)

    sentences = split_sentences(doc)
    lexical_result = lexical_analysis(doc)
    syntactic_result = syntactic_analysis(doc)
    semantic_result = semantic_analysis(doc, syntactic_result)
    visual_prompt = generate_visual_prompt(semantic_result, language)

    image_path = generate_image(visual_prompt)

    data = {
        "original_text": original_text,
        "clean_text": clean_text,
        "language": language,
        "sentences": sentences,
        "lexical_analysis": lexical_result,
        "syntactic_analysis": syntactic_result,
        "semantic_analysis": semantic_result,
        "visual_prompt": visual_prompt,
        "image_path": image_path
    }

    save_results(data)

    return data


def streamlit_app():
    """
    Interfaz gráfica web del Procesador de Lenguaje Natural.
    Esta versión no llama a ninguna API externa de generación de imágenes.
    """

    st.set_page_config(
        page_title="Procesador NLP Text-to-Image",
        page_icon="🧠",
        layout="wide"
    )

    st.title("Procesador de Lenguaje Natural Text-to-Image")

    st.write(
        "Este sistema recibe un texto, realiza análisis léxico, sintáctico "
        "y semántico, y finalmente genera un prompt visual que puede utilizarse "
        "como base para un sistema de generación de imágenes."
    )

    st.info(
        "El objetivo del programa es mostrar paso a paso cómo una oración o "
        "fragmento literario puede transformarse en una representación estructurada "
        "del lenguaje y después en una descripción visual."
    )

    st.sidebar.title("Configuración")

    language_option = st.sidebar.selectbox(
        "Idioma del texto",
        ["Español", "Inglés"]
    )

    if language_option == "Español":
        language = "es"
        example_text = (
            "Un viejo marinero observaba el mar oscuro mientras el barco "
            "avanzaba lentamente entre la niebla."
        )
    else:
        language = "en"
        example_text = (
            "The old sailor watched the dark sea while the ship moved slowly "
            "through the fog."
        )

    st.sidebar.info(
        "Selecciona el idioma del texto para que el programa cargue el modelo "
        "NLP correspondiente. En español se usa un modelo entrenado para español "
        "y en inglés uno entrenado para inglés."
    )

    text = st.text_area(
        "Ingresa el texto a analizar",
        value=example_text,
        height=180
    )

    st.caption(
        "Puedes ingresar una oración simple, una oración compuesta o un fragmento "
        "breve de texto literario. Para mejores resultados, usa fragmentos de "
        "1 a 3 oraciones."
    )

    analyze_button = st.button("Analizar texto", type="primary")

    if analyze_button:
        if not text.strip():
            st.error("Debes ingresar un texto válido.")
            return

        with st.spinner("Analizando el texto..."):
            data = run_pipeline_from_interface(
                original_text=text,
                language=language
            )

        st.success("Análisis finalizado correctamente.")

        # ==============================
        # 1. Texto procesado
        # ==============================

        st.header("1. Texto procesado")

        st.write(
            "En este apartado el programa muestra el texto original ingresado "
            "por el usuario y el texto preprocesado. El preprocesamiento limpia "
            "espacios innecesarios y prepara el contenido para ser analizado."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Texto original")
            st.write(
                "Es el texto tal como fue escrito por el usuario, sin modificaciones."
            )
            st.write(data["original_text"])

        with col2:
            st.subheader("Texto preprocesado")
            st.write(
                "Es la versión limpia del texto. El programa elimina espacios "
                "sobrantes y normaliza la entrada para facilitar el análisis NLP."
            )
            st.write(data["clean_text"])

        # ==============================
        # 2. Oraciones detectadas
        # ==============================

        st.header("2. Oraciones detectadas")

        st.write(
            "En esta etapa el programa divide el texto en oraciones. Esto permite "
            "analizar fragmentos complejos por partes, especialmente cuando el texto "
            "contiene varias acciones o ideas."
        )

        for index, sentence in enumerate(data["sentences"], start=1):
            st.write(f"{index}. {sentence}")

        # ==============================
        # 3. Análisis léxico
        # ==============================

        st.header("3. Análisis léxico")

        st.write(
            "El análisis léxico estudia las unidades básicas del texto. "
            "El programa separa el texto en tokens, identifica el lema de cada palabra, "
            "su categoría gramatical y algunas propiedades útiles para el análisis."
        )

        st.markdown(
            """
            En esta tabla, el programa muestra:

            - **Token:** palabra o signo detectado.
            - **Lemma:** forma base de la palabra.
            - **POS:** categoría gramatical general.
            - **Tag:** etiqueta gramatical más específica.
            - **Dependency:** relación sintáctica con otra palabra.
            - **Head:** palabra de la que depende.
            - **is_alpha:** indica si el token está formado por letras.
            - **is_stop:** indica si es una palabra común o funcional.
            """
        )

        lexical_tokens = data["lexical_analysis"]["tokens"]
        st.dataframe(lexical_tokens, use_container_width=True)

        st.subheader("Entidades nombradas")

        st.write(
            "Las entidades nombradas son nombres propios, lugares, fechas, "
            "organizaciones, personas u otros elementos relevantes reconocidos "
            "automáticamente por el modelo NLP."
        )

        entities = data["lexical_analysis"]["entities"]

        if entities:
            st.dataframe(entities, use_container_width=True)
        else:
            st.info("No se detectaron entidades nombradas.")

        # ==============================
        # 4. Análisis sintáctico
        # ==============================

        st.header("4. Análisis sintáctico")

        st.write(
            "El análisis sintáctico estudia cómo se relacionan las palabras entre sí. "
            "Aquí el programa intenta detectar la estructura de cada oración, "
            "incluyendo verbo principal, sujetos, objetos, acciones y modificadores."
        )

        st.markdown(
            """
            En esta sección, el programa busca:

            - **Verbo raíz:** acción principal de la oración.
            - **Sujetos:** quién o qué realiza la acción.
            - **Objetos:** sobre qué recae la acción.
            - **Acciones:** verbos principales o secundarios.
            - **Modificadores:** palabras que agregan descripción o detalle.
            """
        )

        for index, sentence_data in enumerate(
            data["syntactic_analysis"]["sentences"],
            start=1
        ):
            with st.expander(f"Oración {index}: {sentence_data['sentence']}"):
                st.write("**Verbo raíz:**", sentence_data["root_verb"])
                st.write("**Sujetos:**", sentence_data["subjects"])
                st.write("**Objetos:**", sentence_data["objects"])
                st.write("**Acciones:**", sentence_data["actions"])
                st.write("**Modificadores:**", sentence_data["modifiers"])

        st.subheader("Frases nominales")

        st.write(
            "Las frases nominales son grupos de palabras organizadas alrededor "
            "de un sustantivo. Suelen representar personajes, objetos, conceptos "
            "o elementos importantes del texto."
        )

        noun_phrases = data["syntactic_analysis"]["noun_phrases"]

        if noun_phrases:
            st.dataframe(noun_phrases, use_container_width=True)
        else:
            st.info("No se detectaron frases nominales.")

        st.subheader("Frases preposicionales")

        st.write(
            "Las frases preposicionales suelen indicar lugar, dirección, tiempo, "
            "modo o contexto. Son importantes porque ayudan a ubicar visualmente "
            "la escena."
        )

        prepositional_phrases = data["syntactic_analysis"]["prepositional_phrases"]

        if prepositional_phrases:
            for phrase in prepositional_phrases:
                st.write("-", phrase)
        else:
            st.info("No se detectaron frases preposicionales.")

        # ==============================
        # 5. Análisis semántico
        # ==============================

        st.header("5. Análisis semántico")

        st.write(
            "El análisis semántico interpreta el significado general del texto. "
            "En esta etapa, el programa extrae los elementos más útiles para "
            "construir una escena visual: personajes, acciones, objetos, lugares, "
            "atributos y atmósfera."
        )

        st.info(
            "Este análisis no busca comprender el texto como lo haría una persona, "
            "sino obtener una representación estructurada que permita generar "
            "un prompt visual."
        )

        semantic = data["semantic_analysis"]
        descriptions = get_semantic_descriptions()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personajes o entidades")
            st.write(descriptions["characters_or_entities"])
            st.write(semantic["characters_or_entities"])

            st.subheader("Acciones")
            st.write(descriptions["actions"])
            st.write(semantic["actions"])

            st.subheader("Objetos")
            st.write(descriptions["objects"])
            st.write(semantic["objects"])

        with col2:
            st.subheader("Lugares o contextos")
            st.write(descriptions["places_or_contexts"])
            st.write(semantic["places_or_contexts"])

            st.subheader("Atributos")
            st.write(descriptions["attributes"])
            st.write(semantic["attributes"])

            st.subheader("Atmósfera")
            st.write(descriptions["atmosphere_words"])
            st.write(semantic["atmosphere_words"])

        # ==============================
        # 6. Prompt visual generado
        # ==============================

        st.header("6. Prompt visual generado")

        st.write(
            "Con los datos obtenidos en el análisis semántico, el programa construye "
            "una descripción visual. Esta descripción funciona como un prompt que "
            "podría enviarse a un modelo de generación de imágenes."
        )

        st.code(data["visual_prompt"], language="text")

        # ==============================
        # 7. Resultado visual simulado
        # ==============================

        st.header("7. Resultado visual")

        st.write(
            "En esta versión, el programa no llama a una API de generación de imágenes. "
            "En su lugar, guarda el prompt visual generado en un archivo de texto. "
            "Esto permite simular el resultado final del sistema text-to-image "
            "sin depender de servicios externos."
        )

        st.info(
            "La generación real de imagen puede implementarse posteriormente "
            "con una API externa o un modelo local de generación de imágenes."
        )
      

if __name__ == "__main__":
    streamlit_app()