#!/usr/bin/env python

import argparse
import queue
import sys
import time
from functools import partial

import numpy as np
import os
import requests
import tritonclient.grpc as grpcclient
from PIL import Image
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def curate_log_output(token_sequence,
                      identifier="Input"):
    print(f"{identifier} sequence: ", token_sequence)


def check_output_names(expected_outputs, infer_result):
    if expected_outputs:
        output_names = set([o.name for o in infer_result._result.outputs])
        if set(expected_outputs) != output_names:
            raise Exception(
                f"expected outputs do not match actual outputs {expected_outputs} != {output_names}"
            )


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape,
                              np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def prepare_outputs(output_names):

    outputs = []
    for output_name in output_names:
        outputs.append(grpcclient.InferRequestedOutput(output_name))
    return outputs


def prepare_inputs(text_data, image_data, request_output_len_data,
                   beam_width_data, temperature_data, repetition_penalty_data,
                   presence_penalty_data, streaming_data, 
                   lora_task_id_data, lora_weights_data, lora_config_data,
                   return_log_probs_data, top_k_data, top_p_data):
    inputs = [
        prepare_tensor("text_input", text_data),
        prepare_tensor("image_input", image_data),
        prepare_tensor("max_tokens", request_output_len_data),
        prepare_tensor("beam_width", beam_width_data),
        prepare_tensor("temperature", temperature_data),
        prepare_tensor("stream", streaming_data),
        prepare_tensor("return_log_probs", return_log_probs_data),
        prepare_tensor("top_k", top_k_data),
        prepare_tensor("top_p", top_p_data),
    ]
    if lora_task_id_data is not None:
        inputs += [prepare_tensor("lora_task_id", lora_task_id_data)]
    if lora_weights_data is not None:
        inputs += [
            prepare_tensor("lora_weights", lora_weights_data),
            prepare_tensor("lora_config", lora_config_data),
        ]
    if repetition_penalty_data is not None:
        inputs += [
            prepare_tensor("repetition_penalty", repetition_penalty_data),
        ]
    if presence_penalty_data is not None:
        inputs += [
            prepare_tensor("presence_penalty", presence_penalty_data),
        ]
    return inputs


def prepare_stop_signals():

    inputs = [
        grpcclient.InferInput('text_input', [1, 1], "BYTES"),
        grpcclient.InferInput('max_tokens', [1, 1], "INT32"),
        grpcclient.InferInput('image_input', [1, 3, 224, 224], "FP32"),
        grpcclient.InferInput('stop', [1, 1], "BOOL"),
    ]

    inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.object_))
    inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.zeros([1, 3, 224, 224], dtype=np.float32))
    inputs[3].set_data_from_numpy(np.array([[True]], dtype='bool'))

    return inputs


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        if (FLAGS.streaming):
            if result.get_output('text_output') is not None:
                output = result.as_numpy('text_output')
                print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument('--text',
                        type=str,
                        required=False,
                        default='Question: which city is this? Answer:',
                        help='Input text')

    parser.add_argument(
        '--image',
        type=str,
        required=False,
        default=
        "https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/resolve/main/image1.webp",
        help='Input image')

    parser.add_argument(
        "-s",
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL encrypted channel to the server",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "-r",
        "--root-certificates",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded root certificates. Default is None.",
    )
    parser.add_argument(
        "-p",
        "--private-key",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded private key. Default is None.",
    )
    parser.add_argument(
        "-x",
        "--certificate-chain",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded certificate chain. Default is None.",
    )
    parser.add_argument(
        "-C",
        "--grpc-compression-algorithm",
        type=str,
        required=False,
        default=None,
        help=
        "The compression algorithm to be used when sending request to server. Default is None.",
    )
    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )
    parser.add_argument(
        "-c",
        "--check-output",
        action="store_true",
        required=False,
        default=False,
        help="Enable check of output ids for CI",
    )

    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="temperature value",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        required=False,
        default=1.0,
        help="The repetition penalty value",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        required=False,
        default=None,
        help="The presence penalty value",
    )

    parser.add_argument(
        "--request-output-len",
        type=int,
        required=False,
        default=16,
        help="Request output length",
    )
    parser.add_argument(
        '--stop-after-ms',
        type=int,
        required=False,
        default=0,
        help='Early stop the generation after a few milliseconds')
    parser.add_argument(
        "--stop-via-request-cancel",
        action="store_true",
        required=False,
        default=False,
        help="Early stop use request cancellation instead of stop request")

    parser.add_argument('--request-id',
                        type=str,
                        default='',
                        required=False,
                        help='The request_id for the stop request')

    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32', 'bfloat16'])

    parser.add_argument(
        "--return-log-probs",
        action="store_true",
        required=False,
        default=False,
        help="Enable computation of log probs",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        required=False,
        default=1,
        help="top k value",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        required=False,
        default=0.,
        help="top p value",
    )

    parser.add_argument(
        "--lora-path",
        type=str,
        default='',
        required=False,
        help="LoRA weights"
    )

    parser.add_argument(
        "--lora-task-id",
        type=int,
        default=None,
        required=False,
        help="LoRA task id"
    )

    parser.add_argument(
        "--img-size",
        required=False,
        type=int,
        default=490,
        help="Resized img size value",
    )

    parser.add_argument('--requested-outputs',
                        nargs='+',
                        default=[],
                        help='The requested output tensors')

    FLAGS = parser.parse_args()
    curate_log_output(FLAGS.text, "Input")

    raw_image = Image.open(requests.get(FLAGS.image, stream=True).raw).convert('RGB')
    # raw_image = Image.open(FLAGS.image).convert('RGB')
    vis_processor = transforms.Compose([
        transforms.Resize((FLAGS.img_size, FLAGS.img_size),
                            interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = vis_processor(raw_image).cpu().numpy().astype(np.float16)
    image_data = np.expand_dims(image, axis=0)

    pre_prompt = ''
    meta_instruction = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
    '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
    '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
    '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.',
    pre_prompt += f"""[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
    pre_prompt += f"""[UNUSED_TOKEN_146]user\n"""

    post_prompt = f"""<ImageHere>{FLAGS.text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
    prompt = pre_prompt + post_prompt
    text_data = np.array([[prompt.encode("utf8")]], dtype=np.object_)
    request_output_len = [[FLAGS.request_output_len]]
    request_output_len_data = np.array(request_output_len, dtype=np.int32)
    beam_width = [[FLAGS.beam_width]]
    beam_width_data = np.array(beam_width, dtype=np.int32)
    top_k = [[FLAGS.top_k]]
    top_k_data = np.array(top_k, dtype=np.int32)
    top_p = [[FLAGS.top_p]]
    top_p_data = np.array(top_p, dtype=np.float32)
    temperature = [[FLAGS.temperature]]
    temperature_data = np.array(temperature, dtype=np.float32)
    return_log_probs = [[FLAGS.return_log_probs]]
    return_log_probs_data = np.array(return_log_probs, dtype=bool)

    repetition_penalty_data = None
    if FLAGS.repetition_penalty is not None:
        repetition_penalty = [[FLAGS.repetition_penalty]]
        repetition_penalty_data = np.array(repetition_penalty,
                                           dtype=np.float32)
    presence_penalty_data = None
    if FLAGS.presence_penalty is not None:
        presence_penalty = [[FLAGS.presence_penalty]]
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32)
    streaming = [[FLAGS.streaming]]
    streaming_data = np.array(streaming, dtype=bool)

    lora_weights_data = None
    lora_config_data = None
    if (FLAGS.lora_path != ""):
        lora_weights_data = np.load(
            os.path.join(FLAGS.lora_path, "model.lora_weights.npy"))
        try:
            lora_config_data = np.load(
                os.path.join(FLAGS.lora_path, "model.lora_config.npy"))
        except Exception:
            lora_config_data = np.load(
                os.path.join(FLAGS.lora_path, "model.lora_keys.npy"))
    lora_task_id_data = None
    if FLAGS.lora_task_id is not None and FLAGS.lora_task_id != 0:
        lora_task_id_data = np.array([[FLAGS.lora_task_id]], dtype=np.uint64)

    inputs = prepare_inputs(text_data, image_data, request_output_len_data,
                            beam_width_data, temperature_data,
                            repetition_penalty_data, presence_penalty_data,
                            streaming_data, 
                            lora_task_id_data, lora_weights_data, lora_config_data,
                            return_log_probs_data, top_k_data, top_p_data)

    if FLAGS.requested_outputs:
        # Must have at least output_ids in requested outputs
        if "output_ids" not in FLAGS.requested_outputs:
            raise Exception(
                "requested outputs must at least have \"output_ids\"")
        outputs = prepare_outputs(FLAGS.requested_outputs)
    else:
        outputs = None

    stop_inputs = None
    if FLAGS.stop_after_ms > 0 and not FLAGS.stop_via_request_cancel:
        stop_inputs = prepare_stop_signals()

    request_id = FLAGS.request_id

    cum_log_probs = None
    output_log_probs = None

    user_data = UserData()
    with grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain,
    ) as triton_client:
        try:

            if FLAGS.streaming:

                # Establish stream
                triton_client.start_stream(
                    callback=partial(callback, user_data),
                    stream_timeout=FLAGS.stream_timeout,
                )
                # Send request
                triton_client.async_stream_infer(
                    'ensemble',
                    inputs,
                    outputs=outputs,
                    request_id=request_id,
                )

                if FLAGS.stop_after_ms > 0:
                    time.sleep(FLAGS.stop_after_ms / 1000.0)

                    if not FLAGS.stop_via_request_cancel:
                        triton_client.async_stream_infer(
                            'ensemble',
                            stop_inputs,
                            request_id=request_id,
                            parameters={'Streaming': FLAGS.streaming})

                # Close the grpc stream
                cancel_requests = FLAGS.stop_after_ms > 0 and FLAGS.stop_via_request_cancel
                triton_client.stop_stream(cancel_requests=cancel_requests)

                # Parse the responses
                while True:
                    try:
                        result = user_data._completed_requests.get(block=False)
                    except Exception:
                        break

                    if type(result) == InferenceServerException:
                        if result.status() == "StatusCode.CANCELLED":
                            print("Request is cancelled")
                        else:
                            print("Received an error from server:")
                            print(result)
                            raise result
                    else:
                        check_output_names(FLAGS.requested_outputs, result)
                        output = result.as_numpy('text_output')
                        if output is None:
                            print("Got cancellation response from server")
            else:
                # Send request
                infer_future = triton_client.async_infer(
                    'ensemble',
                    inputs,
                    outputs=outputs,
                    request_id=request_id,
                    callback=partial(callback, user_data),
                    parameters={'Streaming': FLAGS.streaming})

                expected_responses = 1

                if FLAGS.stop_after_ms > 0:

                    time.sleep(FLAGS.stop_after_ms / 1000.0)

                    if FLAGS.stop_via_request_cancel:
                        infer_future.cancel()
                    else:
                        triton_client.async_infer(
                            'ensemble',
                            stop_inputs,
                            request_id=request_id,
                            callback=partial(callback, user_data),
                            parameters={'Streaming': FLAGS.streaming})
                        expected_responses += 1

                processed_count = 0
                while processed_count < expected_responses:
                    try:
                        result = user_data._completed_requests.get()
                        print("Got completed request", flush=True)
                    except Exception:
                        break

                    if type(result) == InferenceServerException:
                        if result.status() == "StatusCode.CANCELLED":
                            print("Request is cancelled")
                        else:
                            print("Received an error from server:")
                            print(result)
                            raise result
                    else:
                        check_output_names(FLAGS.requested_outputs, result)
                        output = result.as_numpy('text_output')
                        if (FLAGS.return_log_probs):
                            cum_log_probs = result.as_numpy('cum_log_probs')
                            output_log_probs = result.as_numpy(
                                'output_log_probs')
                        if output is None:
                            print("Got cancellation response from server")

                    processed_count = processed_count + 1
        except Exception as e:
            err = "Encountered error: " + str(e)
            print(err)
            sys.exit(err)

        for beam in range(FLAGS.beam_width):
            beam_output = output[beam].decode()
            if beam_output:
                curate_log_output(beam_output, "Output")

        if FLAGS.return_log_probs:
            print(cum_log_probs)
            print(output_log_probs)
