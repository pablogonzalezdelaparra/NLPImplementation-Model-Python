from model import NLPModel
from model.Preprocessor import Preprocessor

nlp = NLPModel()
# p = Preprocessor(
#     [
#         "Artificial intelligence, often abbreviated as AI, revolutionizes how we interact with technology. From powering virtual assistants to driving autonomous vehicles, AI systems mimic human intelligence to perform tasks efficiently.",
#         "Its applications span across industries, from healthcare to finance, enhancing productivity and decision-making processes. As AI continues to evolve, its potential to reshape society's dynamics and advance human progress remains unparalleled.",
#     ]
# )

# print()
# # nlp._NLPModel__clean_data('./prueba_temp')

# print(p._Preprocessor__get_text_enum())

# print(p._Preprocessor__tokenize_data())

# print(p._Preprocessor__lower_case())


# print()


nlp.train('./prueba_temp')
