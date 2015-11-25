# coding=utf-8
from dependency_features import DependencyFeatures
# from numpy import sign


class LexFeats(DependencyFeatures):
    def create_arc_features(self,instance,h,m,add=False):
        """ Notes about the code
        - You start by calling the same function, using the parent class. 
          You can build a chain of feature functions in this way.
        - h provides the index of the head word of the dependency arc
        - m provides the index of the modifier word of the dependency arc
        - You can access the part of speech tags in the instance as instance.pos[i], 
          where i indexes any word token.
        - You can access the words themselves as instance.words[i], 
          where i again indexes the token
        - To create a feature, you call getF(), with two arguments:
          - A feature tuple, which includes an index k, and any other information 
            you want -- it need not be a tuple of exactly three items
          - An argument "add", which you don't need to worry about 
            (but you do need to include)
          - Make sure to keep k up-to-date. 
            This prevents collisions in the space of features.
        """
        ff = super(LexFeats,self).create_arc_features(instance,h,m,add)
        k = len(ff)
        f = self.getF((k,instance.pos[h],instance.words[m]),add)
        ff.append(f)
        return ff


# For Deliverable 1a
class LexDistFeats(LexFeats):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(LexDistFeats, self).create_arc_features(instance,h,m,add)
        k = len(ff)
        if abs(h-m) <= 10:
            feat = h-m
        elif h-m < 0:
            feat = -10
        else:
            feat = 10

        f = self.getF((k, feat), add)
        ff.append(f)
        return ff


# For Deliverable 1b
class LexDistFeats2(LexDistFeats):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(LexDistFeats2, self).create_arc_features(instance, h, m, add)
        k = len(ff)
        f = self.getF((k, instance.pos[m], instance.words[h]), add)
        ff.append(f)
        return ff


# For Deliverable 1c
class ContextFeats(LexDistFeats2):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(ContextFeats, self).create_arc_features(instance, h, m, add)
        k = len(ff)
        if h > 1:
            f = self.getF((k, instance.pos[h], instance.pos[h-1], instance.pos[m]), add)   # 49740, 0.800
            ff.append(f)

        if m < len(instance.words)-1:
            f = self.getF((len(ff), instance.pos[h], instance.pos[m], instance.pos[m+1]), add) # 54568, 0.820
            ff.append(f)

        if h > 1 and m < len(instance.words)-1:
            f = self.getF((len(ff), instance.pos[h], instance.pos[h-1], instance.pos[m], instance.pos[m+1]), add) # 54568, 0.820
            ff.append(f)

        return ff


# For delivrable 1d
class BakeoffFeats(ContextFeats):

    def create_arc_features(self,instance,h,m,add=False):
        ff = super(BakeoffFeats, self).create_arc_features(instance, h, m, add)
        # Morphological Features
        # Martha -> laughs
        hdword = self.lookup_word(h)
        dpword = self.lookup_word(h)
        feat = hdword[-1:]    # 0.585
        f = self.getF((len(ff), instance.pos[m], instance.pos[h], feat), add)
        ff.append(f)

        feat = hdword[-2:]    # 0.599
        f = self.getF((len(ff), instance.pos[m], instance.pos[h], feat), add)
        ff.append(f)

        return ff


# For Deliverable 2c
class DelexicalizedFeats(DependencyFeatures):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        # part of speech tag of the head and modifier pair
        ff = super(DelexicalizedFeats, self).create_arc_features(instance,h,m,add)
        k = len(ff)
        f = self.getF((k, instance.pos[m], instance.pos[h]),add)
        ff.append(f)

        # distance between the head and the modifier up to a maximum absolute value of 10
        if abs(h-m) <= 10:
            feat = (h-m)
        elif h-m < 0:
            feat = -10
        else:
            feat = 10
        f = self.getF((len(ff), feat), add)
        ff.append(f)

        return ff

# For Delivrable 2e
# Delexilized + Lexicalized features
class CombinedFeats(DelexicalizedFeats):

    def create_arc_features(self,instance,h,m,add=False):
        ff = super(CombinedFeats, self).create_arc_features(instance, h, m, add)
        f = self.getF((len(ff), instance.pos[m], instance.words[h]), add)
        ff.append(f)
        return ff

# For Delivrable 2f
# Delexilized + Lexicalized + morphological + context features
class CustomFeats(CombinedFeats):

    def create_arc_features(self, instance, h, m, add=False):

        ff = super(CustomFeats, self).create_arc_features(instance, h, m, add)
        # print [self.word_dict.get(word, word) for word in instance.words]

        # Context Features
        if h > 1:
            f = self.getF((len(ff), instance.pos[h], instance.pos[h-1], instance.pos[m]), add)
            ff.append(f)

        if m < len(instance.words)-1:
            f = self.getF((len(ff), instance.pos[h], instance.pos[m], instance.pos[m+1]), add)
            ff.append(f)

        if h > 1 and m < len(instance.words)-1:
            f = self.getF((len(ff), instance.pos[h], instance.pos[h-1], instance.pos[m], instance.pos[m+1]), add)
            ff.append(f)

        # Morphological Features
        # Martha -> laughs
        hdword = self.lookup_word(instance.words[h])
        # dpword = self.lookup_word(instance.words[m])
        feat = hdword[-1:] #58.2
        if feat > '':
            f = self.getF((len(ff), instance.pos[m], instance.pos[h],feat), add)
            ff.append(f)

        feat = hdword[-2:] #0.622
        if feat > '':
            f = self.getF((len(ff), instance.pos[m], instance.pos[h],feat), add)
            ff.append(f)

        return ff


# Section 3
class AdjectifAgreementsFeats(CustomFeats):

    def __init__(self, pos_dict):
        super(AdjectifAgreementsFeats, self)
        self.pos_dict = pos_dict

    def create_arc_features(self,instance,h,m,add=False):

        ff = super(AdjectifAgreementsFeats, self).create_arc_features(instance, h, m, add)
        hdword = self.lookup_word(instance.words[h])
        dpword = self.lookup_word(instance.words[m])

        if instance.pos[h] == self.pos_dict['NOUN'] and instance.pos[m] == self.pos_dict['ADJ']:
            # Number Agreements
            if hdword[-1:] in ['s', 'x'] and dpword[-1:] in ['s', 'x']:
                f = self.getF((len(ff), instance.pos[h], instance.pos[m], '+PL', '+PL'), add)
                ff.append(f)

            # Gender Agreements
            hdgender = self.getGender(hdword)
            dpgender = self.getGender(dpword)
            if dpgender == hdgender:
                f = self.getF((len(ff), instance.pos[h], instance.pos[m], hdgender, dpgender), add)
                ff.append(f)
        return ff

    """
    SOURCE : http://french.about.com/od/grammar/a/genderpatterns.htm
    TODO   : Use OpenFST to create a morphological transducer with these rules. http://pyfst.github.io/
    """
    def getGender(self, word):
        word = word.lower()
        feminineException = ['cage', 'image', 'nage', 'page', 'plage', 'rage', 'cible', 'etable', 'fable', 'table', 'fac', 'boucle',
                    'bride', 'merde', 'méthode', 'pinède', 'eau', 'peau', 'norvège', 'soif', 'clef', 'nef', 'foi', 'loi',
                    'fourmi', 'paroi', 'roseval', 'faim','alarme', 'âme', 'arme', 'cime', 'coutume', 'crème', 'écume', 'énigme',
                    'estime', 'ferme', 'firme', 'forme', 'larme', 'plume', 'rame', 'rime', 'jument', 'façon', 'fin', 'leçon',
                    'main', 'maman', 'rançon', 'dactylo', 'dynamo', 'libido', 'météo', 'moto', 'steno', 'chair', 'cour',
                    'cuiller', 'mer', 'tour','brebis', 'fois', 'oasis', 'souris', 'vis', 'liste', 'modiste', 'piste', 'burlat',
                    'dent', 'dot', 'forêt', 'jument', 'mort','nuit', 'part', 'plupart', 'ziggourat', 'fenêtre', 'huître', 'lettre',
                    'montre', 'rencontre', 'vitre', 'eau', 'peau', 'tribu', 'vertu', 'croix', 'noix', 'paix', 'toux', 'voix']

        masculineException = ['ace', 'palace', 'grade', 'jade', 'stade', 'châle', 'pétale', 'scandale', 'cube', 'cube', 'microbe', 'tube', 'verbe', 'crustacé',
                    'artifice', 'armistice', 'appendice', 'bénéfice', 'caprice', 'commerce', 'dentifrice', 'dentifrice', 'exercice', 'office', 'orifice',
                    'précipice', 'prince', 'sacrifice', 'service', 'silence', 'solstice', 'supplice', 'vice', 'pedigree', 'apogée', 'lycée', 'musée',
                    'périgée', 'trophée', 'bonheur', 'extérieur', 'honneur', 'intérieur', 'malheur', 'meilleur', 'golfe', 'incendie', 'foie',
                    'génie', 'parapluie', 'sosie', 'cimetière', 'cimetière', 'arrière', 'capitaine', 'capitaine', 'moine', 'magazine', 'patrimoine',
                    'avion', 'bastion', 'billion', 'camion', 'cation', 'dominion', 'espion', 'ion', 'lampion', 'lion', 'million', 'morpion',
                    'pion', 'scion', 'scorpion', 'trillion', 'graphique', 'périphérique', 'auditoire', 'commentaire', 'dictionnaire', 'directoire',
                    'horaire', 'itinéraire', 'ivoire', 'laboratoire', 'navire', 'pourboire', 'purgatoire', 'répertoire', 'salaire', 'sommaire', 'sourire',
                    'territoire', 'vocabulaire', 'ermite', 'anthracite', 'granite', 'graphite', 'mérite', 'opposite', 'plébiscite', 'rite', 'satellite', 'site',
                    'termite', 'braille', 'gorille', 'intervalle', 'mille', 'portefeuille', 'vaudeville', 'vermicelle', 'violoncelle', 'dilemme', 'gramme',
                    'programme', 'monde', 'contrôle', 'monopole', 'rôle', 'symbole', 'beurre', 'parterre','tonnerre', 'vestibule', 'centaure', 'cyanure',
                    'verre','carosse', 'colosse', 'gypse','inverse', 'malaise', 'pamplemousse', 'parebrise', 'suspense', 'cyanure', 'murmure',
                    'exposé', 'opposé', 'blason','blouson', 'arrêté', 'comité', 'comté', 'côté', 'député', 'été','pâté',
                    'traité', 'bastion', 'coude', 'interlude', 'prélude', 'abaque', 'préambule', 'scrupule', 'tentacule', 'testicule', 'véhicule', 'ventricule' ]

        if word in feminineException or word[:-1] in feminineException:
            return '+Fem'

        if word in masculineException or word[:-1] in masculineException:
            return '+Masc'

        ############################## Don't move ##################################
        if word[-3:] in ['mme'] or word[-4:] in ['mmes']:
            return '+Fem'

        if word[-3:] in ['ade', 'nde', 'ude', 'ing', 'eur', 'ment', 'son', 'ion' ] or word[-4:] in [ 'ades', 'ndes', 'udes', 'ings', 'eurs', 'ments', 'sons', 'ions']:
            return '+Fem'
        ############################## Don't move ##################################

        if word[-1:] in ['b', 'c', 'd', 'é', 'f', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'x' ]:
            return '+Masc'

        if word[-2:] in ['de', 'et', 'me', 'ou'] or word[-3:] in ['des', 'ets', 'mes', 'oux']:
            return '+Masc'


        if word[-3:] in ['age', 'ble', 'cle', 'eau', 'ège', 'oir', 'one', 'ste', 'tre'] or  word[-4:] in ['ages', 'bles', 'cles', 'eaus', 'èges', 'oirs', 'ones', 'stes', 'tres']:
            return '+Masc'

        if word[-4:] in ['isme'] or word[-5:] in ['ismes']:
            return '+Masc'


        # Feminine
        if word[-2:] in ['be', 'ce', 'cé', 'ee', 'ée', 'fe', 'ie', 'se', 'sé', 'té', 'ue'] or word[-3:] in ['bes', 'ces', 'cés', 'ees', 'ées', 'fes', 'ies', 'ses', 'sés', 'tés', 'ues'] :
            return '+Fem'

        if word[-3:] in ['ace', 'ade', 'ale', 'ine', 'ion', 'ise', 'ire', 'son', 'tié', 'ude', 'ite', 'lle', 'mme', 'nde', 'nne', 'ole','rre', 'ule', 'ure'] or \
                word[-4:] in ['aces', 'ades', 'ales', 'ines', 'ions', 'ises', 'ires', 'sons', 'tiés', 'udes', 'ites', 'lles', 'mmes', 'ndes', 'nnes', 'oles','rres', 'ules', 'ures']:
            return '+Fem'

        if word[-4:] in ['ance', 'esse', 'ière', 'ique', 'sion', 'tion'] or word[-5:] in ['ances', 'esses', 'ières', 'iques', 'sions', 'tions']:
            return '+Fem'

        if word[-1:] in ['e'] or word[-2:] in ['es']:
            return '+Fem'



class DeterminerAgreementsFeats(AdjectifAgreementsFeats):

    def __init__(self, pos_dict):
        super(DeterminerAgreementsFeats, self)
        self.pos_dict = pos_dict

    def create_arc_features(self,instance,h,m,add=False):
        ff = super(DeterminerAgreementsFeats, self).create_arc_features(instance, h, m, add)

        hdword = self.lookup_word(instance.words[h])
        dpword = self.lookup_word(instance.words[m])


        # Enforce distance feature for determinants
        if h-m > 0 and h-m <= 2:
            feat = h-m
        else:
            feat = 0

        if instance.pos[h] == self.pos_dict['NOUN'] and instance.pos[m] == self.pos_dict['DET']:
            if hdword[-1:] in ['s', 'x'] and dpword[-1:] in ['s', 'x']:
                f = self.getF((len(ff), instance.pos[h], instance.pos[m], feat, '+PL', '+PL'), add)
                ff.append(f)

            # Gender Agreements
            hdgender = self.getGender(hdword)
            dpgender = self.getDetGender(dpword)
            if dpgender == hdgender:
                f = self.getF((len(ff), instance.pos[h], instance.pos[m], feat, hdgender, dpgender), add)
                ff.append(f)

        return ff

    def getDetGender(self, word):
        word = word.lower()
        if word in ['le', 'un', 'l', "l\'", 'du', "de l\'", 'ce', 'ces', 'cet', 'quel', 'quels', 'mon', 'ton', 'son', 'aucun', 'ses', 'test', 'aucuns']:
            return '+Masc'

        else:
            return '+Fem'
