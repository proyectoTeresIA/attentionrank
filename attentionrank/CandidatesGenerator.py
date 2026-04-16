
import spacy
from spacy.matcher import Matcher


patterns=[
["NOUN"],
["NOUN","ADJ"],
["NOUN","ADP","NOUN"],
#["VERB"],
["PROPN"],
#["ADJ"],
["PROPN","PROPN"],
#["NOUN","ADP","DET","NOUN"],
["NOUN","ADJ","ADJ"],
#["ADV"],
#["NOUN","ADP","NOUN","ADJ"],
#["PROPN","ADP","PROPN"],
#["PROPN","ADJ"],
#["NOUN","ADP","NOUN","ADP","NOUN"],
#["PROPN","ADP","NOUN"],
#["NOUN","ADJ","PUNCT","ADJ"],
#["NOUN","ADJ","ADP","NOUN"],
#["NOUN","ADJ","PUNCT"],
#["PROPN","PROPN","PROPN"],
#["NOUN","ADP","PROPN"]
]

class CandidatesGenerator():

    def __init__(self,lang):
        self.lang=lang
        if lang == 'es':
            self.nlp = spacy.load("es_core_news_sm")
            print('Spanish Model')
        else:
            self.nlp = spacy.load("en_core_web_sm")
            print('English Model')

    def filtrar_strings(self, lista):
        return [s for s in lista if len(s) > 3]

    def generate_candidates(self, text):

        if self.lang == 'es':
            candidates = self.extract_terms(text,patterns,mode="all")
            return self.filtrar_strings(candidates)

        else:
            candidates = self.extract_terms(text,patterns,mode="all")
            return self.filtrar_strings(candidates)




    def _get_spans(self,doc, pos_patterns):
        """Devuelve todos los spans que cumplen los patrones POS"""
        matcher = Matcher(self.nlp.vocab)

        for i, pat in enumerate(pos_patterns):
            pattern = [{"POS": p} for p in pat]
            matcher.add(f"P{i}", [pattern])

        matches = matcher(doc)

        spans = [doc[start:end] for _, start, end in matches]

        # ordenar por inicio y longitud descendente
        spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
        return spans

    def extract_terms(self,
            text,
            pos_patterns,
            mode="longest",  # "longest", "all", "longest_and_parts"
            lowercase=True,
    ):
        doc = self.nlp(text)

        spans = self._get_spans(doc, pos_patterns)

        # -------------------------
        # MODO ALL
        # -------------------------
        if mode == "all":
            selected = spans

        # -------------------------
        # MODO LONGEST
        # -------------------------
        elif mode == "longest":
            occupied = set()
            selected = []

            for sp in spans:
                token_range = set(range(sp.start, sp.end))

                if not (token_range & occupied):
                    selected.append(sp)
                    occupied.update(token_range)

        # -------------------------
        # LONGEST + PARTS
        # -------------------------
        elif mode == "longest_and_parts":

            # primero obtener longest de verdad (en spans)
            occupied = set()
            longest_spans = []

            for sp in spans:
                token_range = set(range(sp.start, sp.end))
                if not (token_range & occupied):
                    longest_spans.append(sp)
                    occupied.update(token_range)

            # ahora añadir subspans contenidos
            selected = []
            for sp in spans:
                for lg in longest_spans:
                    if sp.start >= lg.start and sp.end <= lg.end:
                        selected.append(sp)
                        break
        else:
            raise ValueError("Modo no válido")

        # -------------------------
        # Formateo salida
        # -------------------------
        terms = []
        for sp in selected:
            t = sp.text.lower() if lowercase else sp.text
            terms.append(t)

        # quitar duplicados preservando orden
        seen = set()
        final = []
        for t in terms:
            if t not in seen:
                final.append(t)
                seen.add(t)

        return final


"""
    
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


"""


#Generator =CandidatesGenerator('es')
#print(Generator.generate_candidates("Trabajo nocturno o laboral, extractivo"))

