from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load the pre-trained model BERT for questions and answers
bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def get_answer_question(context, question):
    # Context and question tokenize
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", add_special_tokens=True)
    #Inference with model BERT ofr receiving the answer
    outputs = bert_model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get answer's start and answer inside the context
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Decode tokens of the answer inside the context
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))

    return answer

def main():
    previous_conversation = ""  # Inicialize prevoius conversation
    
    while True:
        context= previous_conversation  # Context includes previous conversation
        question = input("Tú: ")  # Get user question
        answer = get_answer_question(context, question)
        print("ChatBot:", answer)

        # Actualizar la conversación previa con la pregunta y la respuesta actual
        previous_conversation += f"You: {question}\nChatBot: {answer}\n"

if __name__ == "__main__":
    main()
    