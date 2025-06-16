import spacy
from spacy.language import Language
from spacy.tokens import Token

SPEECH_VERBS = {"say", "tell", "claim", "state", "assert", "remark", "announce", "deny"}
NEGATION_TERMS = {"no", "never", "not", "n't", "none", "nothing", "nowhere", "hardly", "barely", "scarcely", "rarely", "seldom"}

def expand_np(token):
    if token is None:
        return ""
    np_tokens = list(token.subtree)
    for child in token.head.children:
        if child.dep_ in {"appos", "conj", "compound"}:
            np_tokens.extend(list(child.subtree))
    np_tokens = sorted(set(np_tokens), key=lambda x: x.i)
    return " ".join([t.text for t in np_tokens])

class SRLComponent:
    def __init__(self):
        Token.set_extension("srl_arg0", default=None, force=True)
        Token.set_extension("srl_arg1", default=None, force=True)
        Token.set_extension("srl_arg2", default=None, force=True)
        Token.set_extension("srl_negated", default=False, force=True)

    def __call__(self, doc):
        for verb in [t for t in doc if t.pos_ == "VERB"]:
            self.process_verb(verb)
        return doc

    def resolve_subject_control(self, verb):
        current = verb
        visited = set()
        while current.head != current and current not in visited:
            visited.add(current)
            if current.dep_ == "acl" and current.head.pos_ in {"NOUN", "PROPN"}:
                return current.head
            current = current.head
        return None

    def get_first(self, iterable):
        return next(iter(iterable), None)

    def process_verb(self, verb):
        arg0 = self.get_first(t for t in verb.lefts if t.dep_ in {"nsubj", "nsubjpass", "agent"})
        arg1 = self.get_first(t for t in verb.rights if t.dep_ in {"dobj", "attr"})
        arg2 = None

        for prep in verb.children:
            if prep.dep_ == "prep" and prep.lower_ in {"to", "for"}:
                pobj = self.get_first(c for c in prep.children if c.dep_ == "pobj")
                if pobj:
                    arg2 = pobj
                    break

        if verb.tag_ == "VBN" and any(t.dep_ == "auxpass" for t in verb.children):
            arg0, arg1 = arg1, arg0
            if not arg0:
                by_agents = [child for child in verb.children if child.dep_ == "agent"]
                if by_agents:
                    arg0 = self.get_first(tok for tok in by_agents[0].children if tok.dep_ == "pobj")
            if not arg0 and arg1 is not None:
                arg0 = "[Implied observer]"

        if not arg0:
            arg0 = self.resolve_subject_control(verb)

        for child in verb.children:
            if child.dep_ in {"xcomp", "ccomp", "acomp"}:
                if not arg1:
                    arg1 = child
                elif not arg2:
                    arg2 = child

        if verb.dep_ == "relcl" and not arg1:
            head = verb.head
            subject = self.get_first(t for t in head.lefts if t.dep_ in {"nsubj", "nsubjpass"})
            if subject:
                arg1 = subject

        if verb.lemma_ in SPEECH_VERBS:
            speakers = [t for t in verb.lefts if t.dep_ in {"nsubj", "agent"}]
            if not speakers:
                speakers = [t for t in verb.rights if t.dep_ in {"nsubj", "agent"}]
            if speakers:
                arg0 = speakers[0]

        negated = any(t.dep_ == "neg" or t.text.lower() in NEGATION_TERMS for t in verb.children)

        verb._.srl_arg0 = arg0
        verb._.srl_arg1 = arg1
        verb._.srl_arg2 = arg2
        verb._.srl_negated = negated

@Language.factory("srl")
def create_srl_component(nlp, name):
    return SRLComponent()

def get_nlp():
    nlp_local = spacy.load("en_core_web_sm")
    nlp_local.add_pipe("srl", last=True)
    return nlp_local
