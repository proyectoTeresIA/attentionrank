
import spacy
from spacy.matcher import Matcher


patterns_es=[
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

patterns_en=[
    ["NOUN"],
    ["PROPN"],
    ["ADJ"],
    ["VERB"],
    ["ADV"],
    ["NOUN", "NOUN"],
    ["ADJ", "NOUN"],
    ["NOUN", "NOUN", "NOUN"],
    ["ADJ", "NOUN", "NOUN"],
    ["ADJ", "ADJ", "ADJ"],
    ["NOUN", "ADJ"],
    ["ADP", "DET", "NOUN"],
    ["ADP", "NOUN"],
    ["NOUN", "ADP", "NOUN"],
    ["PROPN", "PROPN"],
    ["NOUN", "ADJ", "ADJ"],
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
            candidates = self.extract_terms(text,patterns_es,mode="all")
            return self.filtrar_strings(candidates)

        else:
            candidates = self.extract_terms(text,patterns_en,mode="all")
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




#Generator =CandidatesGenerator('es')
#print(Generator.generate_candidates("Trabajo nocturno o laboral, extractivo"))

