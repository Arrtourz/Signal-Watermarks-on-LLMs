import openai
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import lmppl
from tqdm import tqdm
import logging
from time import sleep
import tiktoken
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import operator

api_key = "YOUR OPENAI API"

encoding = tiktoken.get_encoding("p50k_base")

class gpt_logp:
    """ Language Model. """

    def __init__(self, api_key: str, model: str, sleep_time: int = 10):
        """ Language Model.

        @param api_key: OpenAI API key.
        @param model: OpenAI model.
        """
        logging.info(f'Loading Model: `{model}`')
        openai.api_key = api_key
        self.model = model
        self.sleep_time = sleep_time

    def get_logprobs(self, input_texts: str or list, *args, **kwargs):
        """ Compute the log probabilities and return corresponding tokens on recurrent LM.

        :param input_texts: A string or list of input texts for the encoder.
        :return: A list of tuples, where each tuple contains the log probabilities and corresponding token for a single input text.
        """
        single_input = type(input_texts) == str
        input_texts = [input_texts] if single_input else input_texts
        all_logprobs = []
        for text in tqdm(input_texts):
            while True:
                try:
                    completion = openai.Completion.create(
                        model=self.model,
                        prompt=text,
                        logprobs=5,
                        max_tokens=0,
                        temperature=0,
                        echo=True
                    )
                    break
                except Exception:
                    if self.sleep_time is None or self.sleep_time == 0:
                        logging.exception('OpenAI internal error')
                        exit()
                    logging.info(f'Rate limit exceeded. Waiting for {self.sleep_time} seconds.')
                    sleep(self.sleep_time)
            logprobs = completion['choices'][0]['logprobs']['token_logprobs']
            all_logprobs.append(logprobs)
        return all_logprobs,completion

def generate_text_without_pattern(api_key, prompt, temperature, max_tokens):
    """
    Generate text using OpenAI GPT-3 model without a specific pattern.

    :param api_key: The API key for OpenAI.
    :param prompt: The prompt to send to the model.
    :param temperature: The temperature setting for the model.
    :return: Generated text from the model.
    """
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            logprobs=5,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        model_gen_text = response['choices'][0]['text']
        top_logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
        return model_gen_text
    except Exception as e:
        print(f"Error during API call: {e}")
        return ""

def generate_text_with_pattern(api_key, prompt, pattern, temperature):
    openai.api_key = api_key

    pattern_index = 0
    next_token_rank = 0
    model_gen_text = ""

    for _ in range(len(pattern)):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt + model_gen_text,
            max_tokens=1,
            logprobs=5,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        if response.get('choices') and response['choices'][0].get('logprobs') and response['choices'][0]['logprobs'].get('top_logprobs') and response['choices'][0]['logprobs']['top_logprobs']:
            top_logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]

            if len(top_logprobs) > next_token_rank:
                next_token_rank = pattern[pattern_index % len(pattern)] - 1
                next_token = sorted(top_logprobs.items(), key=lambda x: x[1], reverse=True)[next_token_rank][0]
                model_gen_text += next_token
                pattern_index += 1
            else:
                break
        else:
            break

    return model_gen_text


def analyze_text_watermark(input_text, api_key):
    scorer_gpt35 = gpt_logp(api_key=api_key, model="text-davinci-003")

    logprobs, response_1 = scorer_gpt35.get_logprobs(input_text)

    samples_preview = response_1['choices'][0]['logprobs']['token_logprobs']

    gen_candidates_1 = response_1.choices[0].logprobs.top_logprobs
    gen_candidates_1 = [item for item in gen_candidates_1 if item is not None]

    sorted_gen_candidates_1 = [dict(sorted(item.items(), key=operator.itemgetter(1), reverse=True)) for item in gen_candidates_1]
    gen_candidates_format_1 = [[k for k in d.keys()] for d in sorted_gen_candidates_1]
    tokens_1 = response_1['choices'][0]['logprobs']['tokens']
    tokens_1 = tokens_1[1:]

    rank_output = [gen_candidates_format_1[i].index(tokens_1[i]) + 1 if tokens_1[i] in gen_candidates_format_1[i] else None for i in range(len(tokens_1))]
    rank_outputs = [3 if i is None else i for i in rank_output]
#     rank_outputs = [5 if i is None else i for i in rank_output]
#     rank_outputs = [0 if i is None else i for i in rank_output] #i.e.,None


    return rank_outputs, samples_preview


def generate_pattern(choice, cycles=10, samples_per_cycle=10):
    """
    Generates sine wave patterns with different ranges.
    :param choice: type of mode to choose (1, 2, or 3)
    :param cycles: number of cycles of the sine wave
    :param samples_per_cycle: number of samples per cycle
    :return: Patterns generated based on selection
    """
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

def find_peak_freqs(fft_freq, fft_magnitude, num_peaks=1, freq_threshold=0.05):
    peak_freqs = []

    indices_above_threshold = np.where(fft_freq > freq_threshold)[0]
    if len(indices_above_threshold) > 0:
        filtered_magnitudes = fft_magnitude[indices_above_threshold]
        filtered_freqs = fft_freq[indices_above_threshold]

        peak_indices = np.argsort(filtered_magnitudes)[-num_peaks:]
        for i in peak_indices:
            peak_freq = filtered_freqs[i]
            peak_freqs.append(peak_freq)

    return peak_freqs

def sliding_window_fft_analysis(data, window_size=10, num_peaks=1):
    peak_frequencies_all_windows = []

    if len(data) < window_size:
        window_size = len(data)

    for i in range(len(data) - window_size + 1):
        window_data = data[i:i + window_size]

        fft_result = np.fft.fft(window_data)
        fft_freq = np.fft.fftfreq(window_size)
        fft_magnitude = np.abs(fft_result)

        peak_frequencies = find_peak_freqs(fft_freq, fft_magnitude, num_peaks)
        peak_frequencies_all_windows.append(peak_frequencies)

    return peak_frequencies_all_windows


def find_high_peak_freq_indices(peak_freq_list, threshold=0.1):
    high_peak_indices = []

    for index, peaks in enumerate(peak_freq_list):
        if any(peak_freq > threshold for peak_freq in peaks):
            high_peak_indices.append(index)

    return high_peak_indices


def color_text_console(text, indices_to_color):
    tokens_index = encoding.encode(text)
    colored_text = ""

    for i, token in enumerate(tokens_index):
        if i in indices_to_color:
            colored_token = encoding.decode([token])
        else:
            colored_token = f"\033[31m{encoding.decode([token])}\033[0m"
        colored_text += colored_token

    return colored_text


def mark_peak_freqs_pattern(fft_freq, fft_magnitude, ax, title, num_peaks=3, freq_threshold=0.05):
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

def generate_qpsk_signal(binary_data):
    symbol_mapping = {
        '00': np.pi,       # 180°
        '01': np.pi/2,     # 90°
        '10': 3*np.pi/2,   # 270°
        '11': 0,           # 0°
    }

    carrier_freq = 1
    samples_per_symbol = 100
    total_symbols = len(binary_data) // 2
    x = np.linspace(0, 2 * np.pi * total_symbols, total_symbols * samples_per_symbol)

    qpsk_signal = np.array([])

    for i in range(0, len(binary_data), 2):
        symbol = binary_data[i:i+2]
        phase = symbol_mapping[symbol]
        wave_segment = np.sin(carrier_freq * x[:samples_per_symbol] + phase)
        qpsk_signal = np.concatenate([qpsk_signal, wave_segment])
        x = x[samples_per_symbol:]

    scaled_qpsk_signal = 2 * qpsk_signal + 3

    sample_n = 20
    sampling_interval = 100//sample_n
    sampled_indices = np.arange(0, len(scaled_qpsk_signal), sampling_interval)
    sampled_qpsk_signal = scaled_qpsk_signal[sampled_indices]

    sampled_qpsk_signal_rounded = np.round(sampled_qpsk_signal).astype(int)

    return sampled_qpsk_signal_rounded


def decode_to_ascii(rank_outputs_W, sample_n, window_size=3, sigma=1):
    def gaussian_filter(data, sigma):
        return gaussian_filter1d(data, sigma)

    def sin_wave(x, amplitude, phase, offset):
        return amplitude * np.sin(x + phase) + offset

    if len(rank_outputs_W)>=120:
        data = rank_outputs_W[40:120]
    else:
        s = len(rank_outputs_W)-80
        data = rank_outputs_W[s:]
    subarrays = [data[i:i+20] for i in range(0, len(data), 20)]
    smoothed_data_gaussian = [gaussian_filter(array, sigma) for array in subarrays]

    resampled_subarrays = []
    for original_array, smoothed_array in zip(subarrays, smoothed_data_gaussian):
        if len(smoothed_array) < len(original_array):
            extended_smoothed_array = np.pad(smoothed_array, (0, len(original_array) - len(smoothed_array)), 'edge')
            resampled_subarrays.append(extended_smoothed_array)
        else:
            resampled_subarrays.append(smoothed_array[:len(original_array)])

    x = np.linspace(0, 2 * np.pi, sample_n)
    initial_phases = [0, np.pi/2, np.pi, 3*np.pi/2]
    amplitude_phase_list = []

    for array in resampled_subarrays:
        best_fit_phase = None
        smallest_error = float('inf')
        params = [0, 0, 0]

        for initial_phase in initial_phases:
            try:
                params, _ = curve_fit(sin_wave, x, array, p0=[2, initial_phase, 3])
                fitted_curve = sin_wave(x, *params)
                error = np.sum((fitted_curve - array) ** 2)
                if error < smallest_error:
                    best_fit_phase = params[1]
                    smallest_error = error
            except RuntimeError:
                continue

        amplitude_phase_list.append((params[0], params[1]))

    process_data = []
    for amplitude, phase in amplitude_phase_list:
        if amplitude < 0:
            process_data.append(phase / np.pi - 1)
        else:
            process_data.append(phase / np.pi)

    decode_bin = ''
    for j in process_data:
        if 0.25 < j <= 0.75:
            decode_bin += '01'
        elif 0.75 < j <= 1.25:
            decode_bin += '00'
        elif 1.25 < j <= 1.75:
            decode_bin += '10'
        elif j > 1.75 or 0 < j <= 0.25:
            decode_bin += '11'

    ascii_character = chr(int(decode_bin, 2))
    return ascii_character




# Testing for signal watermark
plt.close('all')
temperature = 0
prompt = "Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. "
real_completion = "He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. He will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information. The cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared."

# sinx pattern plot
pattern = generate_pattern(5, cycles=10, samples_per_cycle=20)
plt.stem(pattern, use_line_collection=True,markerfmt=' ')
plt.xlabel('Token Index')
plt.ylabel('Rank Index')
plt.title('Pattern waveform')
plt.show()

model_gen_text_W = generate_text_with_pattern(api_key, prompt, pattern, temperature)
model_gen_text_NW = generate_text_without_pattern(api_key, prompt, temperature, 200)

rank_outputs_real, samples_preview_real = analyze_text_watermark(real_completion, api_key)
rank_outputs_W, samples_preview_W = analyze_text_watermark(model_gen_text_W, api_key)
rank_outputs_NW, samples_preview_NW = analyze_text_watermark(model_gen_text_NW, api_key)

plt.stem(rank_outputs_real, label='rank in token candidates pool', use_line_collection=True, markerfmt=' ')
plt.plot(samples_preview_real, label='Real Completion', color='C2')
plt.xlabel('Token Index')
plt.ylabel('Log_p Value                      Rank index')
plt.legend()
plt.grid(True)
plt.show()

plt.stem(rank_outputs_NW, label='rank in token candidates pool', use_line_collection=True, markerfmt=' ')
plt.plot(samples_preview_NW, label='No Wateramrk', color='C2')
plt.xlabel('Token Index')
plt.ylabel('Log_p Value                      Rank index')
plt.legend()
plt.grid(True)
plt.show()

plt.stem(rank_outputs_W, label='rank in token candidates pool', use_line_collection=True, markerfmt=' ')
plt.plot(samples_preview_W, label='Wateramrk', color='C2')
plt.xlabel('Token Index')
plt.ylabel('Log_p Value                      Rank index')
plt.legend()
plt.grid(True)
plt.show()

plt.stem(pattern, use_line_collection=True, markerfmt=' ')
plt.xlabel('Token Index')
plt.ylabel('Rank Index')
plt.show()

plt.stem(rank_outputs_real, label='rank in token candidates pool', use_line_collection=True, markerfmt=' ')
plt.plot(samples_preview_real, label='Real Completion', color='C2')
plt.xlabel('Token Index')
plt.ylabel('Log_p Value                      Rank index')
plt.legend()
plt.grid(True)
plt.show()

plt.stem(rank_outputs_NW, label='rank in token candidates pool', use_line_collection=True, markerfmt=' ')
plt.plot(samples_preview_NW, label='No Wateramrk', color='C2')
plt.xlabel('Token Index')
plt.ylabel('Log_p Value                      Rank index')
plt.legend()
plt.grid(True)
plt.show()

plt.stem(rank_outputs_W, label='rank in token candidates pool', use_line_collection=True, markerfmt=' ')
plt.plot(samples_preview_W, label='Wateramrk', color='C2')
plt.xlabel('Token Index')
plt.ylabel('Log_p Value                      Rank index')
plt.legend()
plt.grid(True)
plt.show()

# WM detection
fft_result_pattern = np.fft.fft(pattern)
fft_freq_pattern = np.fft.fftfreq(len(pattern))

fft_result_real = np.fft.fft(rank_outputs_real)
fft_freq_real = np.fft.fftfreq(len(rank_outputs_real))

fft_result_NW = np.fft.fft(rank_outputs_NW)
fft_freq_NW = np.fft.fftfreq(len(rank_outputs_NW))

fft_result_W = np.fft.fft(rank_outputs_W)
fft_freq_W = np.fft.fftfreq(len(rank_outputs_W))
fft_magnitude_pattern = np.abs(fft_result_pattern)
fft_magnitude_real = np.abs(fft_result_real)
fft_magnitude_NW = np.abs(fft_result_NW)
fft_magnitude_W = np.abs(fft_result_W)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("FFT Magnitude Spectrum")

mark_peak_freqs(fft_freq_pattern, fft_magnitude_pattern, axs[0, 0], "Pattern")
mark_peak_freqs(fft_freq_real, fft_magnitude_real, axs[0, 1], "Real Completion")
mark_peak_freqs(fft_freq_NW, fft_magnitude_NW, axs[1, 0], "NW")
mark_peak_freqs(fft_freq_W, fft_magnitude_W, axs[1, 1], "W")

plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust the top spacing to accommodate the main title

plt.show()

peak_freqs_W = find_peak_freqs(fft_freq_W, fft_magnitude_W, num_peaks=1, freq_threshold=0.05)
peak_freqs_pattern = find_peak_freqs(fft_freq_pattern, fft_magnitude_pattern, num_peaks=1, freq_threshold=0.05)

print(peak_freqs_W, peak_freqs_pattern)



# Calculate mixed text percentage, wateramrk detection with slide windows
mix_text_W = prompt+ model_gen_text_W
rank_outputs_W_slide, samples_preview_W_slide = analyze_text_watermark(mix_text_W, api_key)
peak_frequencies_list = sliding_window_fft_analysis(rank_outputs_W_slide)
plt.figure(figsize=(10, 5))
plt.plot(peak_frequencies_list)
plt.xlabel('Slide Windows Index')
plt.ylabel('Frenquency')
high_peak_indices = find_high_peak_freq_indices(peak_frequencies_list)
window_size = 10
indices_lists = [list(range(i, i + window_size)) for i in high_peak_indices]

combined_indices = []
for indices_list in indices_lists:
    combined_indices.extend(indices_list)

combined_indices = sorted(set(combined_indices))

colored_mix_text_W = color_text_console(mix_text_W, combined_indices)
print(colored_mix_text_W)





# testing with sinx,sin2x,sin3x watermark

pattern_0 = generate_pattern(5,20,10)
pattern_1 = generate_pattern(10,20,10)
pattern_2 = generate_pattern(15,20,10)

fig, axes = plt.subplots(3, 1, figsize=(6, 12))

axes[0].stem(pattern_0, use_line_collection=True, markerfmt=' ')
axes[0].set_xlabel('Token Index')
axes[0].set_ylabel('Rank index')
axes[0].set_title('Pattern 0 in sin(x) waveform')

axes[1].stem(pattern_1, use_line_collection=True, markerfmt=' ')
axes[1].set_xlabel('Token Index')
axes[1].set_ylabel('Rank index')
axes[1].set_title('Pattern 1 in sin(2x) waveform')

axes[2].stem(pattern_2, use_line_collection=True, markerfmt=' ')
axes[2].set_xlabel('Token Index')
axes[2].set_ylabel('Rank index')
axes[2].set_title('Pattern 2 in sin(3x) waveform')

plt.tight_layout()
plt.show()

model_gen_text_W0 = generate_text_with_pattern(api_key, prompt, pattern_0, temperature)
model_gen_text_W1 = generate_text_with_pattern(api_key, prompt, pattern_1, temperature)
model_gen_text_W2 = generate_text_with_pattern(api_key, prompt, pattern_2, temperature)

rank_outputs_W0, samples_preview_W0 = analyze_text_watermark(model_gen_text_W0, api_key)
rank_outputs_W1, samples_preview_W1 = analyze_text_watermark(model_gen_text_W1, api_key)
rank_outputs_W2, samples_preview_W2 = analyze_text_watermark(model_gen_text_W2, api_key)

plt.stem(rank_outputs_W0, label='rank in top_logprobs pool',use_line_collection=True, markerfmt=' ')
plt.plot(samples_preview_W0, label='sinx watermark',color='C2')
plt.xlabel('Token Index')
plt.ylabel('Log_p Value                      Rank index')
plt.legend()
plt.grid(True)
plt.show()

plt.stem(rank_outputs_W1, label='rank in top_logprobs pool',use_line_collection=True, markerfmt=' ')
plt.plot(samples_preview_W1, label='sin2x watermark',color='C2')
plt.xlabel('Token Index')
plt.ylabel('Log_p Value                      Rank index')
plt.legend()
plt.grid(True)
plt.show()

plt.stem(rank_outputs_W2, label='rank in top_logprobs pool',use_line_collection=True, markerfmt=' ')
plt.plot(samples_preview_W2, label='sin3x watermark',color='C2')
plt.xlabel('Token Index')
plt.ylabel('Log_p Value                      Rank index')
plt.legend()
plt.grid(True)
plt.show()


cycles = 10
samples_per_cycle = 10
x = np.linspace(0, 2 * np.pi * cycles, cycles * samples_per_cycle)
sin_wave = np.sin(x)+np.sin(2*x)+np.sin(3*x)
scaled_wave = (sin_wave + 1)*2
pattern = np.round(scaled_wave).astype(int) + 1

fft_result_pattern = np.fft.fft(pattern)
fft_freq_pattern = np.fft.fftfreq(len(pattern))

fft_result_W0 = np.fft.fft(rank_outputs_W0)
fft_freq_W0 = np.fft.fftfreq(len(rank_outputs_W0))

fft_result_W1 = np.fft.fft(rank_outputs_W1)
fft_freq_W1 = np.fft.fftfreq(len(rank_outputs_W1))

fft_result_W2 = np.fft.fft(rank_outputs_W2)
fft_freq_W2 = np.fft.fftfreq(len(rank_outputs_W2))

fft_magnitude_pattern = np.abs(fft_result_pattern)
fft_magnitude_W0 = np.abs(fft_result_W0)
fft_magnitude_W1 = np.abs(fft_result_W1)
fft_magnitude_W2 = np.abs(fft_result_W2)

plt.close('all')
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("FFT Magnitude Spectrum")

mark_peak_freqs_pattern(fft_freq_pattern, fft_magnitude_pattern, axs[0, 0], "Pattern")
mark_peak_freqs(fft_freq_W0, fft_magnitude_W0, axs[0, 1], "W_sinx")
mark_peak_freqs(fft_freq_W1, fft_magnitude_W1, axs[1, 0], "W_sin2x")
mark_peak_freqs(fft_freq_W2, fft_magnitude_W2, axs[1, 1], "W_sin3x")
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust the top spacing to accommodate the main title
plt.show()




# testing on C4 dataset
dataset = load_dataset("c4", "realnewslike", split="train")
temperature = 0
NW=[]
W=[]
real=[]
result_real=[]
result_W=[]
result_NW=[]
PPL_res=[]

pattern = generate_pattern(5, cycles=10, samples_per_cycle=10)
fft_result_pattern = np.fft.fft(pattern)
fft_freq_pattern = np.fft.fftfreq(len(pattern))
fft_magnitude_pattern = np.abs(fft_result_pattern)
peak_freqs_pattern = find_peak_freqs(fft_freq_pattern, fft_magnitude_pattern, num_peaks=1, freq_threshold=0.05)

for i in range(1000):
    all_index=encoding.encode(dataset[i]['text'])
    if len(all_index) <= 250:
        continue
    prompt_index=all_index[0:-200]
    real_completion_index=all_index[-200:]
    prompt=encoding.decode(prompt_index)
    real_completion=encoding.decode(real_completion_index)
    real.append(real_completion)

    model_gen_text_W = generate_text_with_pattern(api_key, prompt, pattern, temperature)
    W.append(model_gen_text_W)
    model_gen_text_NW = generate_text_without_pattern(api_key, prompt, temperature, 200)
    NW.append(model_gen_text_NW)
    
    rank_outputs_real, samples_preview_real = analyze_text_watermark(real_completion, api_key)
    rank_outputs_W, samples_preview_W = analyze_text_watermark(model_gen_text_W, api_key)
    rank_outputs_NW, samples_preview_NW = analyze_text_watermark(model_gen_text_NW, api_key)
    

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
    peak_freqs_NW = find_peak_freqs(fft_freq_NW, fft_magnitude_NW, num_peaks=1, freq_threshold=0.05)
    peak_freqs_W = find_peak_freqs(fft_freq_W, fft_magnitude_W, num_peaks=1, freq_threshold=0.05)
    
    delta0 = peak_freqs_real[0]-peak_freqs_pattern[0]
    delta1 = peak_freqs_W[0]-peak_freqs_pattern[0]
    delta2 = peak_freqs_NW[0]-peak_freqs_pattern[0]
    
    print(delta0, delta1, delta2)
    result_real.append(delta0)   
    result_W.append(delta1)
    result_NW.append(delta2)


print(real)
print(NW)
print(W)

print(result_real)
print(result_W)
print(result_NW)

# calculate PPL
scorer = lmppl.OpenAI(api_key=api_key, model="text-davinci-003")
scores = scorer.get_perplexity([result_W])

# QPSK to encode message in LLMs
binary_data_list = []

for char in range(ord('A'), ord('Z') + 1):
    binary_data = format(char, '08b')
    binary_data_list.append(binary_data)

for char in range(ord('a'), ord('z') + 1):
    binary_data = format(char, '08b')
    binary_data_list.append(binary_data)

for char in range(ord('0'), ord('9') + 1):
    binary_data = format(char, '08b')
    binary_data_list.append(binary_data)

print(binary_data_list)
test_res = []
pre = np.ones(40, dtype=int)


for i in binary_data_list:
    sampled_qpsk_signal = generate_qpsk_signal(i)
    pattern = np.concatenate([pre, sampled_qpsk_signal])
    model_gen_text_W = generate_text_with_pattern(api_key, prompt, pattern, temperature)
    print(model_gen_text_W)
    rank_outputs_W, samples_preview_W = analyze_text_watermark(model_gen_text_W, api_key)
    ascii_character = decode_to_ascii(rank_outputs_W, sample_n=20)
    test_res.append(ascii_character)
    print("Decoded ASCII Character:", ascii_character)

