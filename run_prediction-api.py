import os
import json
import argparse

from dataclasses import asdict
from tqdm import tqdm

from typing import Any, Dict, Optional

import requests


from src import utils, dataset_loader

#API_URL = "http://0.0.0.0:8001/model-inference"
API_URL = "http://165.132.77.57:8001/model-inference"
DEFAULT_USER_ID = "demo-user"
DEFAULT_SESSION_ID = "working-agent-session"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# =========================
# 공통 유틸 함수
# =========================

def is_anthropic_model(model_id: str) -> bool:
    return "anthropic." in model_id

def is_llama_like_model(model_id: str) -> bool:
    return "meta." in model_id

def is_gemma_like_model(model_id: str) -> bool:
    # return model_id.contains("google.gemma")
    return "google." in model_id



def extract_answer(result):
    """
    Bedrock 응답에서 실제 답변 텍스트를 뽑아내는 함수.
    - 우리가 아래에서 만들어주는 dict 형태를 기준으로 파싱.
    - 혹시 다른 구조가 들어와도 최대한 안전하게 처리.
    """
    if result is None:
        return ""

    if isinstance(result, dict):
        # ChatCompletion 유사 구조
        choices = result.get("choices")
        if choices and len(choices) > 0:
            choice = choices[0]
            # chat 형식
            if isinstance(choice, dict):
                msg = choice.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                # text completion 형식
                text = choice.get("text")
                if isinstance(text, str):
                    return text

        # fallback: "answer" 키가 있을 경우
        ans = result.get("answer")
        if isinstance(ans, str):
            return ans

    return ""


def chat_completion_to_json(chat_completion):
    """
    예전에는 dataclass ChatCompletion을 asdict로 변환했겠지만,
    지금은 dict로 가정하고 그대로 dump.
    """
    return json.dumps(chat_completion, ensure_ascii=False, indent=4)



def _invoke_chat(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    *,
    api_url: str = API_URL,
    user_id: str = DEFAULT_USER_ID,
    session_id: str = DEFAULT_SESSION_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    use_model_server: bool = True,
    timeout: int = 120,
    extra_invoke_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calls your custom inference API and normalizes the response
    into an OpenAI-ish wrapper (choices/usage/finish_reason etc).
    """

    invoke_kwargs = {
        "temperature": float(temperature),
        "max_new_tokens": int(max_tokens),
    }
    # 필요하면 여기에 top_p, repetition_penalty 등 추가 가능
    if extra_invoke_kwargs:
        invoke_kwargs.update(extra_invoke_kwargs)

    body = {
        "user_id": user_id,
        "session_id": session_id,
        "model": {
            "model": model_name,
            "use_model_server": bool(use_model_server),
        },
        "prompt": prompt,
        "invoke_kwargs": invoke_kwargs,
    }

    try:
        resp = requests.post(api_url, json=body)
        resp.raise_for_status()
    except requests.RequestException as e:
        # 네트워크/HTTP 에러를 wrapper 형태로 반환(원하면 raise 해도 됨)
        return {
            "id": "",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "error",
                }
            ],
            "usage": {"input_tokens": None, "output_tokens": None},
            "raw_api_response": {"error": str(e)},
        }

    response_body = resp.json()

    # --- Your API response parsing ---
    # Example response:
    # {
    #   "steps": [...],
    #   "model_metadata": {
    #       "model": "...",
    #       "endpoint": "...",
    #       "finish_reason": "stop",
    #       "usage": {"prompt_tokens": 15, "completion_tokens": 24, "total_tokens": 39}
    #   },
    #   "completion": "....",
    #   "session": {}
    # }
    answer_text = response_body.get("completion", "")
    model_meta = response_body.get("model_metadata", {}) or {}
    usage_meta = model_meta.get("usage", {}) or {}

    input_tokens = usage_meta.get("prompt_tokens")
    output_tokens = usage_meta.get("completion_tokens")
    finish_reason = model_meta.get("finish_reason", "stop")

    # --- Normalize to OpenAI-ish wrapper (기존 코드가 이 포맷을 기대한다면 유용) ---
    wrapped = {
        "id": response_body.get("id", ""),  # 없으면 빈값
        "model": model_meta.get("model", model_name),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_text,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage_meta.get("total_tokens"),
        },
        "raw_api_response": response_body,  # 디버깅용
    }

    return wrapped



def _invoke_complete(prompt: str, max_tokens: int = 1024, temperature: float = 0.0):
    chat_like = _invoke_chat(prompt, max_tokens=max_tokens, temperature=temperature)

    answer_text = extract_answer(chat_like)

    # completion 스타일 흉내
    if chat_like.get("choices") and isinstance(chat_like["choices"][0], dict):
        chat_like["choices"][0]["text"] = answer_text
    else:
        chat_like["choices"] = [{"index": 0, "text": answer_text}]

    return chat_like



def multi_threading_running(fn, context_list, n_workers: int = 1):
    """
    멀티스레딩 비활성화: 항상 순차 실행.
    ThreadPoolExecutor 오버헤드를 제거해서 latency를 줄임.
    """
    results = []
    for ctx in context_list:
        results.append(fn(ctx))
    return results


# =========================
# 기존 openai_* API 대체
# =========================
def query_openai(context_list, setting_name, n_multiply=2):
    if setting_name == 'complete':
        fn = _invoke_complete
    else:
        fn = _invoke_chat

    # 항상 순차 실행
    results = multi_threading_running(fn, context_list, n_workers=1)
    return results




def query_openai_with_retry(context_list, setting_name, retry_time=4, results=None):
    """
    Bedrock 버전 retry 로직. extract_answer(result)가 빈 문자열이면 재시도.
    """
    if results is None:
        results = query_openai(context_list, setting_name)

    while retry_time > 0:
        filtered_context_list = []
        idx_map = []  # 원래 index를 기록

        for i in range(len(results)):
            if extract_answer(results[i]) == "":
                filtered_context_list.append(context_list[i])
                idx_map.append(i)

        if len(filtered_context_list) == 0:
            # print("nothing need to retry")
            break

        filtered_results = query_openai(filtered_context_list, setting_name)
        print("filtered_results in query_openai_with_retry:", filtered_results)

        # 결과를 원래 위치에 다시 채워 넣기
        for p, idx in enumerate(idx_map):
            results[idx] = filtered_results[p]

        # retry 성과 로깅
        retry_succeeded = 0
        for item in filtered_results:
            if extract_answer(item) != "":
                retry_succeeded += 1
        print(
            "In the retry, {0} samples succeeded, {1} samples failed".format(
                retry_succeeded, len(filtered_results) - retry_succeeded
            )
        )
        if retry_succeeded <= 3:
            retry_time -= 1

    assert len(results) == len(context_list)
    return results


# =========================
# 데이터셋 관련 함수
# =========================
def run_multiple_dataset_batch(work_items):
    if len(work_items) == 0:
        return
    dataset_list = []
    item_list = []
    for (input_path, output_path, mode, _) in work_items:
        assert mode == work_items[0][2]
        js_list = utils.read_jsonl(input_path)
        content_list = [item["context"] for item in js_list]
        dataset_list.append(len(content_list))
        item_list += content_list

    # Bedrock 호출
    raw_results = query_openai_with_retry(context_list=item_list, setting_name=work_items[0][2])

    # answer + input_tokens + output_tokens 형태로 변환
    formatted_results = [format_result_with_tokens(r) for r in raw_results]

    print("results:", formatted_results)

    # dataset 단위로 나누어 저장
    s = 0
    for i in range(len(dataset_list)):
        utils.save_jsonl(formatted_results[s: s + dataset_list[i]], work_items[i][1])
        s += dataset_list[i]
    assert s == len(formatted_results)

    # # debugging용 확인용
    # global is_debug
    # s = 0
    # for i in range(len(dataset_list)):
    #     save_dir = work_items[i][1]
    #     if is_debug:
    #         save_dir = save_dir.replace('.jsonl', '-debug.jsonl')
    #     utils.save_jsonl(formatted_results[s: s + dataset_list[i]], save_dir)
    #     s += dataset_list[i]
    # assert s == len(formatted_results)



def run_multiple_dataset(work_items):
    batch = []
    count = 0
    # Bedrock에서도 너무 크게는 안 묶는 게 안전하므로 500 정도 유지
    # batch_size = 500
    batch_size = 1

    for item in tqdm(work_items, desc='Running dataset...'):
        if os.path.exists(item[1]):
            if len(utils.read_jsonl(item[1])) == item[3]:
                continue
        if count + item[3] > batch_size:
            run_multiple_dataset_batch(batch)
            batch = []
            count = 0
        batch.append(item)
        count += item[3]
    if len(batch) > 0:
        run_multiple_dataset_batch(batch)


def format_result_with_tokens(result):
    answer = extract_answer(result)
    usage = result.get("usage", {})
    return {
        "answer": answer,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }

def parse_args():
    p = argparse.ArgumentParser()

    # 핵심: dataset / setting
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Dataset names. e.g., lsat-ar lsat-lr")
    p.add_argument("--settings", nargs="+", default=["zero-shot"],
                   help="Setting names. e.g., zero-shot zero-shot-CoT few-shot")

    # 경로/옵션들
    p.add_argument("--dataset_dir", type=str, default="data/v1_1")
    p.add_argument("--raw_prompt_path", type=str, default="./data/few_shot_prompts.csv")
    p.add_argument("--output_dir", type=str, default="outputs/qwen4b")
    p.add_argument("--chat_mode", action="store_true", help="Use chat_mode", default=True)
    p.add_argument("--max_tokens", type=int, default=2048)

    # stage control
    p.add_argument("--skip_stage_1", action="store_true", default=False)
    p.add_argument("--skip_stage_2", action="store_true", default=True)
    p.add_argument("--skip_stage_3", action="store_true", default=True)

    # debug
    p.add_argument("--debug", action="store_true", default=False)

    return p.parse_args()

# =========================
# 메인 실행부
# =========================
if __name__ == "__main__":

    args = parse_args()

    dataset_dir = args.dataset_dir
    raw_prompt_path = args.raw_prompt_path
    output_dir = args.output_dir

    skip_stage_1 = args.skip_stage_1
    skip_stage_2 = args.skip_stage_2
    skip_stage_3 = args.skip_stage_3
    # is_debug = args.debug
    is_debug = True

    chat_mode = args.chat_mode
    max_tokens = args.max_tokens


    run_experiment = True
    # dataset_dir = "data/v1_1"
    # raw_prompt_path = "./data/few_shot_prompts.csv"

    # # 출력 디렉토리 및 모델 이름 (파일명에만 사용)

    # if DEFAULT_MODEL_NAME == "Qwen/Qwen3-4B-Instruct-2507":
    #     output_dir = "outputs/qwen4b"
    # else:
    #     raise KeyError()
    gpt_model = "local"  # 단순 파일명용 문자열

    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    # dataset_name_list = ["sat-math",]
    # dataset_name_list = [
    #     # "sat-math",

    #     "lsat-ar",
    #     # "lsat-lr", 
    #     # "lsat-rc",

    #     # "gaokao-geography",
    #     # "gaokao-history",
    #     # "gaokao-biology",
    #     # "gaokao-chemistry",
    #     # "gaokao-physics",
    #     # "gaokao-mathqa",
    #     # "gaokao-english",
    #     # "gaokao-chinese",

    #     # "sat-en",
    #     # "aqua-rat",

    #     # "logiqa-en", 
    #     # "logiqa-zh",
    #     # "gaokao-mathcloze",
    #     # "jec-qa-kd", 
    #     # "jec-qa-ca",
    #     # "math",
    #     # "sat-en-without-passage",
    # ]
    # setting_name_list = [
    #     # 'few-shot',
    #     # 'few-shot-CoT',
    #     'zero-shot',
    #     # 'zero-shot-CoT'
    # ]

    dataset_name_list = args.datasets if args.datasets is not None else [
        # "sat-math",

        # "lsat-ar",
        # "lsat-lr", 
        # "lsat-rc",

        # "gaokao-geography",
        # "gaokao-history",
        # "gaokao-biology",
        # "gaokao-chemistry",
        # "gaokao-physics",
        # "gaokao-mathqa",
        # "gaokao-english",
        # "gaokao-chinese",

        # "sat-en",
        # "aqua-rat",

        "logiqa-en", 
        # "logiqa-zh",
        # "gaokao-mathcloze",
        # "jec-qa-kd", 
        # "jec-qa-ca",
        # "math",
        # "sat-en-without-passage",
    ]
    setting_name_list = args.settings


    # skip_stage_1 = False
    # skip_stage_2 = False
    # skip_stage_3 = True
    
    # is_debug = False

    # chat_mode = True

    work_items = []
    for dataset_name in dataset_name_list:
        for setting_name in setting_name_list:
            dataset = dataset_loader.load_dataset(
                dataset_name, setting_name, dataset_dir,
                prompt_path=raw_prompt_path, max_tokens=2048,
                end_of_example="<END>\n", verbose=True, chat_mode=chat_mode)
            if is_debug:
                dataset = dataset[:2]
            input_path = os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.jsonl")
            utils.save_jsonl(dataset, input_path)
            output_path = os.path.join(
                output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.jsonl')
            first_stage_output_path = os.path.join(
                output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.first_stage.jsonl')

            if 'few-shot' in setting_name:
                work_items.append((input_path, output_path, 'chat' if chat_mode else 'complete', len(dataset)))
            else:
                work_items.append((input_path, first_stage_output_path, 'chat', len(dataset)))

    if not skip_stage_1:
        print('='*50)
        print(f'Stage 1 :\n\twork_items:{work_items[0]}')
        run_multiple_dataset([item for item in work_items if item[2] == 'complete'])
        run_multiple_dataset([item for item in work_items if item[2] == 'chat'])

    work_items = []
    for dataset_name in dataset_name_list:
        for setting_name in setting_name_list:
            if 'few-shot' in setting_name:
                continue
            dataset = dataset_loader.load_dataset(
                dataset_name, setting_name, dataset_dir,
                prompt_path=raw_prompt_path, max_tokens=2048,
                end_of_example="<END>\n", verbose=True)
            if is_debug:
                dataset = dataset[:2]
            input_path = os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.jsonl")
            output_path = os.path.join(
                output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.jsonl')
            first_stage_output_path = os.path.join(
                output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.first_stage.jsonl')

            first_stage_results = utils.read_jsonl(first_stage_output_path)
            second_stage_input = dataset_loader.generate_second_stage_input(
                dataset_name, dataset, first_stage_results)
            second_stage_input_path = os.path.join(
                output_dir, "inputs", f"{dataset_name}.{setting_name}.second_stage.jsonl")
            utils.save_jsonl(second_stage_input, second_stage_input_path)
            work_items.append((second_stage_input_path, output_path, 'chat', len(dataset)))

    if not skip_stage_2:
        print('='*50)
        print(f'Stage 2 :\n\twork_items:{work_items[0]}')
        run_multiple_dataset(work_items)

    if not skip_stage_3:
        wrong_dataset_name_setting_name_list = [
            ("aqua-rat", "few-shot-CoT"),
            ("math", "few-shot"),
            ("math", "few-shot-CoT"),
            ("gaokao-physics", "few-shot-CoT"),
        ]
        for dataset_name, setting_name in wrong_dataset_name_setting_name_list:
            zero_shot_dataset = dataset_loader.load_dataset(
                dataset_name, "zero-shot", dataset_dir,
                prompt_path=raw_prompt_path, max_tokens=2048,
                end_of_example="<END>\n", verbose=True)
            few_shot_output_path = os.path.join(
                output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.jsonl')
            few_shot_second_stage_output_path = os.path.join(
                output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.second_stage.jsonl')
