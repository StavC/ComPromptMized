import os
from io import BytesIO

import requests
import torchvision.transforms as T
from PIL import Image
from FlowSteeringWorm.llava.conversation import conv_templates
from FlowSteeringWorm.llava.model import *
from transformers import AutoTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor

transform = T.ToPILImage()
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(42)
from transformers import logging

logging.set_verbosity_error()

SEED = 10
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

TEMPERATURE = 0.1
MAX_NEW_TOKENS = 1024
CONTEXT_LEN = 2048

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def generate_stream(model, prompt, tokenizer, input_ids, images=None):
    temperature = TEMPERATURE
    max_new_tokens = MAX_NEW_TOKENS
    context_len = CONTEXT_LEN
    max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]
    stop_idx = 2

    ori_prompt = prompt
    image_args = {"images": images}

    output_ids = list(input_ids)
    pred_ids = []

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    past_key_values = None

    for i in range(max_new_tokens):
        if i == 0 and past_key_values is None:
            out = model(
                torch.as_tensor([input_ids]).cuda(),
                use_cache=True,
                output_hidden_states=True,
                **image_args,
            )
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device="cuda"
            )
            out = model(
                input_ids=torch.as_tensor([[token]], device="cuda"),
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            logits = out.logits
            past_key_values = out.past_key_values
        # yield out

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)
        pred_ids.append(token)

        if stop_idx is not None and token == stop_idx:
            stopped = True
        elif token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i != 0 and i % 1024 == 0 or i == max_new_tokens - 1 or stopped:
            cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
            pos = -1  # cur_out.rfind(stop_str)
            if pos != -1:
                cur_out = cur_out[:pos]
                stopped = True
            output = ori_prompt + cur_out

            # print('output', output)

            ret = {
                "text": output,
                "error_code": 0,
            }
            yield cur_out

        if stopped:
            break

    if past_key_values is not None:
        del past_key_values


def run_result(X, prompt, initial_query, query_list, model, tokenizer, unnorm, image_processor):
    device = 'cuda'
    X = load_image(X)

    print("Image: ")
    # load the image
    X = image_processor.preprocess(X, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().cuda()

    # Generate the output with initial query
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device=device)

    res = generate_stream(model, prompt, tokenizer, input_ids[0].tolist(), X)
    for response1 in res:
        outputs1 = response1

    print(f'Query 1:')
    print(initial_query)
    print(f'Response 1:')
    print(outputs1.strip())

    print('********')
    ALLResponses = []
    ALLResponses.append(outputs1.strip())

    # Generate the outputs with further queries
    for idx, query in enumerate(query_list):
        if idx == 0:
            # Update current prompt with the initial prompt and first output
            new_prompt = prompt + outputs1 + "\n###Human: " + query + "\n###Assistant:"

        else:
            # Update current prompt with the previous prompt and latest output
            new_prompt = (
                    new_prompt + outputs + "\n###Human: " + query + "\n###Assistant:"
            )

        input_ids = tokenizer.encode(new_prompt, return_tensors="pt").cuda()

        # Generate the response using the updated prompt
        res = generate_stream(model, new_prompt, tokenizer, input_ids[0].tolist(), X)
        for response in res:
            outputs = response

        # Print the current query and response
        print(f"Query {idx + 2}:")
        print(query)
        print(f"Response {idx + 2}:")
        print(outputs.strip())

        print("********")
        ALLResponses.append(outputs.strip())
    return ALLResponses


def Turn_On_LLaVa():  # Load the LLaVa model
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"

    torch.cuda.set_device(0)
    device = torch.device('cuda')
    print('Current Device :', torch.cuda.current_device())
    MODEL_NAME = "FlowSteering/llava/llava_weights/"  # PATH to the LLaVA weights
    model_name = os.path.expanduser(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtypePerDevice = torch.float16

    model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=dtypePerDevice,
                                                  use_cache=True)
    model.to(device=device, dtype=dtypePerDevice)
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=dtypePerDevice,
                                                   low_cpu_mem_usage=True)
    model.to(device=device, dtype=dtypePerDevice)
    model.get_model().vision_tower[0] = vision_tower
    vision_tower.to(device=device, dtype=dtypePerDevice)

    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    return model, image_processor, tokenizer, device


def load_param(MODEL_NAME, model, tokenizer, initial_query):
    model_name = os.path.expanduser(MODEL_NAME)

    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    vision_tower = model.get_model().vision_tower[0]
    vision_tower = CLIPVisionModel.from_pretrained(
        vision_tower.config._name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.get_model().vision_tower[0] = vision_tower

    if vision_tower.device.type == "meta":
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device="cuda", dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    unnorm = UnNormalize(image_processor.image_mean, image_processor.image_std)
    norm = Normalize(image_processor.image_mean, image_processor.image_std)

    embeds = model.model.embed_tokens.cuda()
    projector = model.model.mm_projector.cuda()

    for param in vision_tower.parameters():
        param.requires_grad = False

    for param in model.parameters():
        param.requires_grad = False

    for param in projector.parameters():
        param.requires_grad = False

    for param in embeds.parameters():
        param.requires_grad = False

    for param in model.model.parameters():
        param.requires_grad = False

    qs = initial_query
    if mm_use_im_start_end:
        qs = (
                qs
                + "\n"
                + DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                + DEFAULT_IM_END_TOKEN
        )
    else:
        qs = qs + "\n" + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    if conv_mode is not None and conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    return (
        tokenizer,
        image_processor,
        vision_tower,
        unnorm,
        norm,
        embeds,
        projector,
        prompt,
        input_ids,
    )


def Run_LLaVa(X, prompt, initial_query, query_list, model, tokenizer, unnorm, image_processor):
    reply = run_result(X, prompt, initial_query, query_list, model, tokenizer, unnorm, image_processor)
    return reply
