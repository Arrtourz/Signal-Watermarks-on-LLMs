import random
from transformers import T5Tokenizer
import tiktoken
import torch
import re
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
encoding = tiktoken.get_encoding("p50k_base")


text =" This event, sponsored locally in partnership with the Montana Food & Drink Alliance is sure not to disappoint! You can sign in to attend at the Missoual Community Culntery. \n\nAt Lestars' class you will learn the secrets and science for smoking, seasoning, and griling your food with ease! Tony will be sharing the techniques used at championship level competitions and will provide hands one learning for everyone attending the class. You can look up some amazing reciped and tips on his"
tokens_ind = encoding.encode(text)

percentage = 10

num_tokens_to_replace = max(1, len(tokens_ind) * percentage // 100)

indices_to_replace = random.sample(range(len(tokens_ind)), num_tokens_to_replace)
res=''
x=0
true_label=[]

for i, index in enumerate(tokens_ind):
    if i in indices_to_replace:
        res+=f"<extra_id_{x}>"
        x+=1
        true_label.append(0)
    else:
        res+=encoding.decode([index])
        true_label.append(1)




# Initialize T5
T5_PATH = 't5-large'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

# Text with multiple placeholders
text = res# Add more <extra_id_{i}> as needed.

# Encode the text
encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
input_ids = encoded['input_ids'].to(DEVICE)

# Calculate the maximum extra_id token present in the text
extra_ids = re.findall(r'<extra_id_(\d+)>', text)
max_extra_id = max([int(id_) for id_ in extra_ids]) if extra_ids else -1

# Generate predictions
outputs = t5_model.generate(
    input_ids=input_ids,
    num_beams=10,
    num_return_sequences=1,
    max_length=len(input_ids[0]) + 50  # You may want to increase this if more output is needed
)

def _filter(output):
    # Decode the full sequence
    full_decoded = t5_tokenizer.decode(output, skip_special_tokens=False)
    
    # Initialize result with the original text
    result = text
    
    # Replace each placeholder with its corresponding generated text
    for i in range(max_extra_id + 1):
        token = f'<extra_id_{i}>'
        pattern = re.escape(token) + r'(.+?)(?=<extra_id_\d+>|</s>)'
        
        # Search for the pattern in the full decoded text
        match = re.search(pattern, full_decoded)
        
        # If there's a match, replace the placeholder with the matched group
        if match:
            result = re.sub(re.escape(token), match.group(1).strip(), result, count=1)
    
    # Clean up any remaining extra_id tokens that were not replaced
    result = re.sub(r'<extra_id_\d+>', '', result).strip()
    
    return result

# Apply the _filter function to generate the final results
results = [_filter(output) for output in outputs]

# Print or return the results
for result in results:
    print(result)


print(true_label)

# # paraphrase
model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')
sentence_to_paraphrase = " This event, sponsored locally in partnership with the Montana Food & Drink Alliance is sure not to disappoint! You can sign in to attend at the Missoual Community Culntery. \n\nAt Lestars' class you will learn the secrets and science for smoking, seasoning, and griling your food with ease! Tony will be sharing the techniques used at championship level competitions and will provide hands one learning for everyone attending the class. You can look up some amazing reciped and tips on his"

input_text = "paraphrase: " + sentence_to_paraphrase
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(input_ids, num_return_sequences=1, num_beams=5, max_length=50, early_stopping=True)
for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))



