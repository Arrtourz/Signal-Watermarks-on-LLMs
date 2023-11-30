
from transformers import OPTForCausalLM, GPT2Tokenizer,pipeline
import torch
import torch
import lmppl
from datasets import load_dataset

# model_name = "facebook/opt-66b"
# model_name = "facebook/opt-iml-max-30b"
model_name = "facebook/opt-1.3b"
# model_name = "facebook/opt-6.7b"
# model_name = "facebook/opt-350m"
# prompt = "Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. "
# real_completion = "He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. He will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information. The cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared."

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = OPTForCausalLM.from_pretrained(model_name).to(device)
model.eval()

def generate_text_without_pattern(prompt, model_name=model_name, max_length=100, temperature=1):
    prompt_length = len(tokenizer.encode(prompt))
    
    if temperature <= 0:
        temperature = 1e-10
    generator = pipeline('text-generation', model=model_name, device=0, do_sample=True, max_length=max_length+prompt_length, temperature=temperature)

    generated_texts = generator(prompt)

    output_text = ""
    for generated in generated_texts:
        generated_tokens = tokenizer.encode(generated["generated_text"])
        output_tokens = generated_tokens[prompt_length:]

        output_text += tokenizer.decode(output_tokens) + "\n"

    return output_text.strip()


def generate_pattern(choice, cycles=10, samples_per_cycle=10):

    x = np.linspace(0, 2 * np.pi * cycles, cycles * samples_per_cycle)
    
    if choice == 5:
        sin_wave = np.sin(x)
        scaled_wave = (sin_wave + 1) * 2
        pattern = np.round(scaled_wave).astype(int) + 1
    elif choice == 10:
        sin_wave = np.sin(2*x)
        scaled_wave = (sin_wave + 1) * 2
        pattern = np.round(scaled_wave).astype(int) + 1
    elif choice == 15:
        sin_wave = np.sin(3*x)
        scaled_wave = (sin_wave + 1) * 2
        pattern = np.round(scaled_wave).astype(int) + 1
    elif choice == 3:
        sin_wave = np.sin(x)
        scaled_wave = sin_wave + 2
        pattern = np.round(scaled_wave).astype(int)
    elif choice == 2:
        sin_wave = np.sin(x)
        scaled_wave = (sin_wave / 2) + 1.5
        pattern = np.round(scaled_wave).astype(int)
    else:
        raise ValueError("Invalid choice. Please select 1, 2, or 3.")
    
    return pattern

def generate_text_with_pattern(prompt, tokenizer, model, pattern=[3, 4, 5], max_candidates=5, temperature=1.0):
    device = model.device  # 获取模型的设备
    model_gen_text = ""
    pattern_index = 0

    for _ in range(len(pattern)):
        inputs = tokenizer(prompt + model_gen_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits[0, -1]

        if temperature <= 0:
            temperature = 1e-10
        scaled_logits = logits / temperature

        probs = torch.softmax(scaled_logits, dim=-1)
        topk_probs, topk_indices = probs.topk(max_candidates)
        next_token_rank = pattern[pattern_index % len(pattern)] - 1

        next_token_id = topk_indices[next_token_rank].item()
        next_token = tokenizer.decode([next_token_id])

        while '</s>' in next_token:
            next_token_rank = (next_token_rank + 1) % max_candidates
            next_token_id = topk_indices[next_token_rank].item()
            next_token = tokenizer.decode([next_token_id])

        model_gen_text += next_token
        pattern_index += 1

    return model_gen_text


def generate_next_token_with_opt(prompt, tokenizer, model, max_candidates=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    probs = torch.softmax(logits, dim=-1)

    topk_probs, topk_indices = probs.topk(max_candidates)
    tokens = [tokenizer.decode([idx]) for idx in topk_indices]

    log_probs = torch.log(topk_probs)

    # 清理
    del inputs, outputs, logits, probs, topk_probs, topk_indices

    return list(zip(tokens, log_probs.tolist())), probs.tolist()



def calculate_token_ranks(generated_text, tokenizer, model, max_candidates=5):
    tokenized_text = tokenizer.encode(generated_text)
    rank_outputs = []

    for i in range(1, len(tokenized_text)):
        current_input_ids = tokenized_text[:i]
        inputs = torch.tensor([current_input_ids]).to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=inputs)

        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        next_token_id = tokenized_text[i]

        topk_probs, topk_indices = probs.topk(max_candidates)
        if next_token_id in topk_indices:
            prob_next_token = probs[next_token_id].item()
            rank = (probs > prob_next_token).sum().item()
            rank = min(rank + 1, max_candidates)
        else:
            rank = max_candidates

        rank_outputs.append(rank)

    return rank_outputs


def mark_peak_freqs(fft_freq, fft_magnitude, ax, title, num_peaks=1, freq_threshold=0.05):
    indices_above_threshold = np.where(fft_freq > freq_threshold)[0]
    if len(indices_above_threshold) > 0:
        filtered_magnitudes = fft_magnitude[indices_above_threshold]
        filtered_freqs = fft_freq[indices_above_threshold]

        peak_indices = np.argsort(filtered_magnitudes)[-num_peaks:]
        for i in peak_indices:
            peak_freq = filtered_freqs[i]
            peak_magnitude = filtered_magnitudes[i]
            ax.semilogy(peak_freq, peak_magnitude, 'ro')
            ax.text(peak_freq, peak_magnitude, f' {peak_freq:.2f} Hz', verticalalignment='bottom')

    ax.semilogy(fft_freq, fft_magnitude)
    ax.set_title(title)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude (Log Scale)')


def find_peak_freqs(fft_freq, fft_magnitude, num_peaks=1, freq_threshold=0.05):
    peak_freqs = []
    # Filter frequencies near the DC component
    indices_above_threshold = np.where(fft_freq > freq_threshold)[0]
    if len(indices_above_threshold) > 0:
        filtered_magnitudes = fft_magnitude[indices_above_threshold]
        filtered_freqs = fft_freq[indices_above_threshold]

        # Find the largest peaks
        peak_indices = np.argsort(filtered_magnitudes)[-num_peaks:]
        for i in peak_indices:
            peak_freq = filtered_freqs[i]
            peak_magnitude = filtered_magnitudes[i]
            peak_freqs.append(peak_freq)

    return peak_freqs


dataset = load_dataset("c4", "realnewslike", split="train")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name)
pattern = generate_pattern(5, cycles=10, samples_per_cycle=20)

temperature = 0.1
fft_result_pattern = np.fft.fft(pattern)
fft_freq_pattern = np.fft.fftfreq(len(pattern)) 
fft_magnitude_pattern = np.abs(fft_result_pattern)
peak_freqs_pattern = find_peak_freqs(fft_freq_pattern, fft_magnitude_pattern, num_peaks=1, freq_threshold=0.05)

NW=[]
W=[]
real=[]
result_real=[]
result_W=[]
result_NW=[]



for i in range(100):
    input_ids = tokenizer.encode(dataset[i]['text'], return_tensors='pt')
    
    if input_ids.shape[1] <= 250 or input_ids.shape[1]>=550:
        continue
    prompt_ids = input_ids[:, :-200]
    prompt = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
    real_completion = tokenizer.decode(input_ids[0, -200:], skip_special_tokens=True)
    real.append(real_completion)
    
    opt_text_NW = generate_text_without_pattern(prompt, temperature=temperature, max_length=200)
    opt_text_W = generate_text_with_pattern(prompt,tokenizer, model, pattern=pattern, temperature=temperature)
    NW.append(opt_text_NW)
    W.append(opt_text_W)
    
    rank_outputs_W = calculate_token_ranks(opt_text_W, tokenizer, model)
    rank_outputs_real = calculate_token_ranks(real_completion, tokenizer, model)
    rank_outputs_NW = calculate_token_ranks(opt_text_NW, tokenizer, model)

    fft_result_real = np.fft.fft(rank_outputs_real)
    fft_freq_real = np.fft.fftfreq(len(rank_outputs_real))

    fft_result_NW = np.fft.fft(rank_outputs_NW)
    fft_freq_NW = np.fft.fftfreq(len(rank_outputs_NW))

    fft_result_W = np.fft.fft(rank_outputs_W)
    fft_freq_W = np.fft.fftfreq(len(rank_outputs_W))


    fft_magnitude_real = np.abs(fft_result_real)
    fft_magnitude_NW = np.abs(fft_result_NW)
    fft_magnitude_W = np.abs(fft_result_W)
    
    peak_freqs_real = find_peak_freqs(fft_freq_real, fft_magnitude_real, num_peaks=1, freq_threshold=0.05)
    peak_freqs_W = find_peak_freqs(fft_freq_W, fft_magnitude_W, num_peaks=1, freq_threshold=0.05)
    peak_freqs_NW = find_peak_freqs(fft_freq_NW, fft_magnitude_NW, num_peaks=1, freq_threshold=0.05)
    
    delta0 = peak_freqs_real[0]-peak_freqs_pattern[0]
    delta1 = peak_freqs_W[0]-peak_freqs_pattern[0]
    delta2 = peak_freqs_NW[0]-peak_freqs_pattern[0]
    
    print(delta0,delta1,delta2)
    result_real.append(delta0)
    result_W.append(delta1)
    result_NW.append(delta2)


print(result_real)
print(result_W)
print(result_NW)

print(real)
print(W)
print(NW)
