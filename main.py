from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st


model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cpu()

if __name__ == '__main__':

    title = st.text_input('your text', '')

    if st.button('click'):

        input_ids = tokenizer.encode(title, return_tensors="pt").cpu()
        out = model.generate(input_ids.cpu())
        generated_text = list(map(tokenizer.decode, out))[0]
        st.write(generated_text)