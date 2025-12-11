
import spacy



class CandidatesGenerator():

    def __init__(self,lang):
        self.lang=lang
        if lang == 'es':
            self.nlp = spacy.load("es_core_news_sm")
            print('Spanish Model')
        else:
            self.nlp = spacy.load("en_core_web_sm")
            print('English Model')


    def generate_candidates(self, text):

        if self.lang=='es':
            candidates= self.__generate_candidates_es(text)
            return candidates

        else:
            candidates = self.__generate_candidates_en(text)
            return candidates
    def remove_starting_articles(self,text):
            # Lista de artículos a eliminar

            if self.lang == 'es':
                articles = ['se ', 'la ', 'el ', 'un ', 'una ', 'unos ', 'unas ', 'los ', 'las ', 'esta ', 'este ',
                            'estos ', 'estas ', 'cada ']
            else:
                articles = ['a ', 'the ', 'an ', 'this ', 'those ', 'that ', 'which ', 'every ']
            text_low = text

            # Iterar sobre cada artículo
            for article in articles:
                # Si el texto comienza con el artículo, quitarlo
                if text_low.lower().startswith(article):
                    text = text[len(article):]  # Quitar el artículo

            return text

    def __generate_candidates_es(self,text):
        candidates = []
        doc = self.nlp(text)
        lis = []
        for chunk in doc.noun_chunks:
            if len(chunk.text) < 2:
                continue
            lis.append((chunk.start_char, chunk.end_char))
        for a, b in lis:
            # print(a,b)
            tokens = []
            labels = []
            for token in doc:
                # print(token.idx)
                if a <= token.idx <= b:
                    # print(token.text, "-", token.pos_,token.idx)
                    tokens.append(token.text)
                    labels.append(token.pos_)

            candidate = self.__construct_candidates_es(tokens, labels)
            if len(candidate) > 1:
                candidates.append(candidate)
        return candidates






    def __construct_candidates_es(self, tokens, labels):

        candiate = ''
        essential = {'NOUN', 'PROPN'}
        possible = {'NOUN', 'ADJ', 'ADV', 'PROPN'}
        internals = {'DET', 'ADJ', 'ADV', 'ADP', 'PROPN', 'NOUN'}
        essentialbool = False

        for tok, lab in zip(tokens, labels):
            if lab in essential:
                candiate = candiate + tok + ' '
                essentialbool = True
                continue

            if lab in possible:
                candiate = candiate + tok + ' '
                continue
            if lab in internals and essentialbool:
                candiate = candiate + tok + ' '
            if lab not in internals and essentialbool:
                break
        return candiate.strip()

    def __generate_candidates_en(self, text):
        candidates = []
        doc = self.nlp(text)
        for chunk in doc.noun_chunks:
            # chunk.root.dep_, chunk.root.head.text)
            chunk_processed = self.remove_starting_articles(chunk.text.lower())
            # chunk_processed = chunk_processed.lower()
            if len(chunk_processed) < 2:
                continue
            candidates.append(chunk_processed)




#Generator =CandidatesGenerator('es')
#print(Generator.generate_candidates("Trabajo nocturno o laboral, extractivo"))

