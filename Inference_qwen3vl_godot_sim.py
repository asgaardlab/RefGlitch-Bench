import os
import json
import base64
from openai import OpenAI
from tqdm import tqdm

from unsloth import FastVisionModel # FastLanguageModel for LLMs
import json
from PIL import Image

# New imports for similarity retrieval
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",  #lora_qwen3vl_8b_godot_caption
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
)

FastVisionModel.for_inference(model)

# ============================================================
# New: frozen image embedding model for similarity search
# ============================================================

sim_model_name = "facebook/dinov2-base"#"facebook/dinov3-vitb16-pretrain-lvd1689m"

sim_processor = AutoImageProcessor.from_pretrained(sim_model_name)
sim_model = AutoModel.from_pretrained(sim_model_name).to("cuda")
sim_model.eval()


@torch.no_grad()
def get_image_embedding(image_path):
    """
    Extract one normalized image embedding for similarity search.
    """
    img = Image.open(image_path).convert("RGB")

    inputs = sim_processor(
        images=img,
        return_tensors="pt"
    ).to("cuda")

    outputs = sim_model(**inputs)

    # CLS token as global image representation
    emb = outputs.last_hidden_state[:, 0]

    # Normalize so dot product = cosine similarity
    emb = F.normalize(emb, dim=-1)

    return emb.cpu()


def find_most_similar_previous_frame(current_emb, previous_embs):
    """
    current_emb: [1, d]
    previous_embs: list of [1, d] tensors

    Return:
        index of the most similar previous frame
    """
    previous_embs = torch.cat(previous_embs, dim=0)  # [i, d]
    sims = current_emb @ previous_embs.T             # [1, i]
    ref_idx = torch.argmax(sims, dim=1).item()
    return ref_idx



def parse_json_response(response):
    text = response.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    text = text[start:]
    if "}" in text:
        text = text[:text.rfind("}") + 1]
    else:
        # Try to close a truncated JSON object
        text = text + "}"

    return text


def call_qwen(tokenizer, message, log_file, test_frame, ref_frame):

    input_text = tokenizer.apply_chat_template(message, add_generation_prompt = True)

    test_img = Image.open(test_frame).convert("RGB")
    if ref_frame is not None:
        ref_img = Image.open(ref_frame).convert("RGB")
    

        inputs = tokenizer(
            images = [ref_img, test_img],
            text = input_text,
            add_special_tokens = False,
            truncation = False,  # avoid dropping image tokens
            return_tensors = "pt",
        ).to("cuda")
    else:
        inputs = tokenizer(
            images = test_img,
            text = input_text,
            add_special_tokens = False,
            truncation = False,  # avoid dropping image tokens
            return_tensors = "pt",
        ).to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    generated_ids = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 10240,
                    use_cache = True, temperature = 0.1, do_sample = False) #temperature = 1.5, min_p = 0.1

    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens = True)
    print('message: ', message)
    print('Response: ', response)
    response = parse_json_response(response)
    predict_label = 1 if json.loads(response)['glitch_detected'] else 0

    # write response to log_file
    with open(log_file, 'a') as f:
        f.write(
            f'Video: {test_frame}, Ref: {ref_frame}, Response: {response}, Prediction: {predict_label}\n')
    return predict_label



file = open('glitch_with_ref_godot', 'r')
context_question_1 = file.read()
file.close()

file = open('glitch_without_ref_godot', 'r')
context_question_2 = file.read()
file.close()

file = open('glitch_with_ref_buggy_godot', 'r')
context_question_3 = file.read()
file.close()

log_file = "qwen3vl_8b_most_similar_previous_godot.txt"
log_file_idx = 'qwen3vl_8b_most_similar_previous_godot_selected.txt'
res_dict = {}

folders_path = "godot_video_folders.txt"
with open(folders_path, "r", encoding="utf-8") as f:
    video_folders = [line.strip() for line in f if line.strip()]
idx = 0
for folder in tqdm(video_folders):
    idx += 1
    if folder.startswith('.'):
        continue
    # if idx < 800:
    #     continue
    print('Processing folder: ', folder, idx)
    file_list = sorted(os.listdir(folder))
    
    # New: store embeddings and predicted labels for previous frames
    emb_list = []
    pred_list = []

    for i in range(len(file_list)):
        file = file_list[i]
        test_frame = os.path.join(folder, file)
        # New: compute embedding for the current frame
        current_emb = get_image_embedding(test_frame)

        if i==0:
            message = [
                {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": context_question_2}
                ]}
        ]

            res = call_qwen(tokenizer, message, log_file, test_frame, None)
        else:
            # ============================================================
            # Main change:
            # Instead of using i-1, retrieve most similar previous frame.
            # ============================================================

            ref_idx = find_most_similar_previous_frame(
                current_emb=current_emb,
                previous_embs=emb_list,
            )
            with open(log_file_idx, "a") as f:
                f.write(f"Selected ref index: {ref_idx}, Current index: {i}\n")

            ref_file = file_list[ref_idx]
            ref_frame = os.path.join(folder, ref_file)

            # Important:
            # Since ref_frame is not necessarily i-1 anymore,
            # use pred_list[ref_idx], not just the previous variable `res`.
            ref_pred = pred_list[ref_idx]

            if ref_pred == 0:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": context_question_1},
                        ],
                    }
                ]
                res = call_qwen(tokenizer, message, log_file, test_frame, ref_frame)

            else:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": context_question_3},
                        ],
                    }
                ]
                res = call_qwen(tokenizer, message, log_file, test_frame, ref_frame)

        # New: after processing current frame, save its embedding and prediction
        emb_list.append(current_emb)
        pred_list.append(res)












