from utils import get_search_index, generate_answer, set_model_and_embeddings, get_question_type

def index():
    set_model_and_embeddings()
    get_search_index()
    return True

def run(question):
    index()
    # return generate_answer(question)
    return get_question_type(question)
