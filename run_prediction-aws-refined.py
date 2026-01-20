# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# import boto3  # AWS Bedrock용
from src import utils, dataset_loader

# # =========================
# # AWS Bedrock 설정 부분
# # =========================
# # ⚠️ 여기를 환경에 맞게 수정하세요.
# BEDROCK_REGION = "us-east-1"
# BEDROCK_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"
# bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# session = boto3.Session()
# creds = session.get_credentials()

# print(f'creds.access_key:{creds.access_key}')
# print(f'creds.secret_key:{creds.secret_key}')

# =========================
# local 실행용
# =========================
import requests
import time

# ==== End Point Info ====
# Gemma / Llama: http://165.132.80.39:8001/model-inference
# Qwen: http://165.132.77.57:8001/model-inference
# ==== Model Info ====
# gemma: google/gemma-3-27b-it  
# llamma: meta-llama/Llama-3.1-8B-Instruct
# Qwen: Qwen/Qwen3-4B-Instruct-2507

ENDPOINT = os.getenv("OMNIA_ENDPOINT", "http://165.132.77.57:8001/model-inference")
MODEL_NAME = os.getenv("OMNIA_MODEL", "Qwen/Qwen3-4B-Instruct-2507")  
DEFAULT_USER_ID = os.getenv("OMNIA_USER_ID", "demo-user")
DEFAULT_SESSION_ID = os.getenv("OMNIA_SESSION_ID", "agieval-local")

# =========================
# 랜덤 샘플링 함수 0112
# =========================
import random

def take_random_subset(dataset, n, seed=42):
    """
    seed 고정 랜덤 샘플링
    - 재현성 보장
    - stage 1 / stage 2에서 동일 subset 유지 가능
    """
    if n is None or n <= 0 or n >= len(dataset):
        return dataset

    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), n)
    idxs.sort()  # 🔴 중요: 원래 순서 유지 (stage2 매칭 안정)
    return [dataset[i] for i in idxs]

# =========================
# 로깅 0112
# =========================
import logging
import threading
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(
    LOG_DIR,
    f"latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),  # 콘솔에도 같이 보고 싶으면 유지
    ],
)

LATENCIES = []
LAT_LOCK = threading.Lock()

logging.info(f"Logging to {LOG_PATH}")


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


# =========================
# 추가 로직 0111
# =========================
import hashlib
import time
import json
import os

DEBUG_DUMP_DIR = os.getenv("OMNIA_DEBUG_DIR", "debug_payloads")
os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)

def _dump_payload(payload, tag=""):
    # prompt가 너무 길 수 있으니 파일로 저장 (콘솔에 다 찍으면 터짐)
    prompt = payload.get("prompt", "")
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]
    ts = int(time.time())
    path = os.path.join(DEBUG_DUMP_DIR, f"{ts}_{tag}_{h}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# =========================
# Bedrock 호출 래퍼 (Local)
# =========================
def _bedrock_invoke_chat(prompt: str, max_tokens: int = 1024, temperature: float = 0.0):
    payload = {
        "user_id": DEFAULT_USER_ID,
        "session_id": DEFAULT_SESSION_ID,
        "model": {
            "model": MODEL_NAME,
            "use_model_server": True,
        },
        "prompt": prompt,
        "invoke_kwargs": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
        },
    }

    # ===== [추가_0111] 요청 직전 최소 정보 로그 =====
    print(f"[invoke] model={MODEL_NAME} prompt_len={len(prompt)} max_new_tokens={max_tokens}")

    # 전체 payload를 파일로 저장 (필요하면 항상 저장, 또는 조건부 저장)
    dump_path = _dump_payload(payload, tag="before")
    print(f"[dump] payload saved: {dump_path}")

    t0 = time.time()

    # =====

    resp = requests.post(
        ENDPOINT,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=(60, 100),
    )

    # ===== [추가_0111] 에러면 응답 본문도 같이 출력+저장
    if resp.status_code >= 400:
        print("[HTTP ERROR]", resp.status_code, resp.text[:1000])
        err_path = os.path.join(DEBUG_DUMP_DIR, f"error_resp_{int(time.time())}.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"[dump] error response saved: {err_path}")
    # =====


    resp.raise_for_status()
    response_body = resp.json()

    # ===== [추가_0112]
    # sssprint(f"[done] {time.time()-t0:.1f}s finish_reason={(response_body.get('model_metadata', {}) or {}).get('finish_reason')}")
    elapsed = time.time() - t0
    finish_reason = (response_body.get("model_metadata", {}) or {}).get("finish_reason")

    with LAT_LOCK:
        LATENCIES.append(elapsed)

    logging.info(
        f"done | latency={elapsed:.2f}s | prompt_len={len(prompt)} | finish_reason={finish_reason}"
    )
    # =====

    # OMNIA 응답 포맷: {"completion": "...", "model_metadata": {"usage": {...}} ...}
    answer_text = response_body.get("completion", "")

    usage = (response_body.get("model_metadata", {}) or {}).get("usage", {}) or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    wrapped = {
        "id": "",
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer_text},
                "finish_reason": (response_body.get("model_metadata", {}) or {}).get("finish_reason", "stop"),
            }
        ],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "raw_response": response_body,
    }
    return wrapped


def _bedrock_invoke_complete(prompt: str, max_tokens: int = 1024, temperature: float = 0.0):
    # complete도 결국 chat처럼 보내고 text를 채워서 리턴(기존 로직 유지)
    chat_like = _bedrock_invoke_chat(prompt, max_tokens=max_tokens, temperature=temperature)
    answer_text = extract_answer(chat_like)
    if chat_like.get("choices") and isinstance(chat_like["choices"][0], dict):
        chat_like["choices"][0]["text"] = answer_text
    else:
        chat_like["choices"] = [{"index": 0, "text": answer_text}]
    return chat_like



def multi_threading_running(fn, context_list, n_workers: int):
    """
    기존 openai_api.multi_threading_running을 대체하는 멀티스레딩 유틸.
    fn: single context를 받아 응답을 리턴하는 함수
    context_list: 각 샘플의 context 문자열 리스트
    n_workers: 최대 동시 스레드 수
    """
    n_workers = max(1, min(n_workers, len(context_list)))
    results = [None] * len(context_list)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(fn, ctx): idx
            for idx, ctx in enumerate(context_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error on index {idx}: {e}")
                # 실패 시 빈 답변으로 채워서 retry 대상이 되게 한다.
                results[idx] = {
                    "error": str(e),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "text": "",
                        }
                    ],
                }
    return results


# =========================
# 기존 openai_* API 대체
# =========================
def query_openai(context_list, setting_name, n_multiply=2):
    """
    원래 Azure OpenAI를 호출하던 함수를 Bedrock용으로 재구현.
    setting_name: 'complete' 또는 'chat'
    context_list: 각 샘플의 prompt/context 문자열 리스트
    n_multiply: 예전 동작을 흉내내기 위한 병렬 계수 (대략적인 worker 수로 사용)
    """
    # 적당한 기본 worker 수 계산 (너무 많이 열리지 않도록 제한)
    base_workers = 2  # 필요시 조정, 기본은 4
    n_workers = max(1, min(len(context_list), base_workers * n_multiply))
    # print("multi-thread workers =", n_workers)

    if setting_name == 'complete':
        fn = _bedrock_invoke_complete
    else:
        fn = _bedrock_invoke_chat

    results = multi_threading_running(fn, context_list, n_workers)
    return results


def query_openai_with_retry(context_list, setting_name, retry_time=1, results=None):
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

    # print("results:", formatted_results)

    # dataset 단위로 나누어 저장
    s = 0
    for i in range(len(dataset_list)):
        utils.save_jsonl(formatted_results[s: s + dataset_list[i]], work_items[i][1])
        s += dataset_list[i]
    assert s == len(formatted_results)



def run_multiple_dataset(work_items):
    batch = []
    count = 0
    # Bedrock에서도 너무 크게는 안 묶는 게 안전하므로 500 정도 유지
    batch_size = 500

    for item in work_items:
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


# =========================
# 메인 실행부
# =========================
SUBSET_N = 50        # ← 50개만 돌리고 싶을 때
SUBSET_SEED = 42     # ← seed 고정

if __name__ == "__main__":
    run_experiment = True
    dataset_dir = "data/v1_1"
    raw_prompt_path = "./data/few_shot_prompts.csv"

    # 출력 디렉토리 및 모델 이름 (파일명에만 사용)
    output_dir = f"outputs/{MODEL_NAME}"
    gpt_model = "local"  # 단순 파일명용 문자열

    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    #dataset_name_list = ["sat-math",]
    dataset_name_list = [

        # "sat-math",
        # "gaokao-geography",
        # "gaokao-history",
        # "gaokao-biology",
        # "gaokao-chemistry",
        # "gaokao-physics",
        # "gaokao-mathqa",
        # "gaokao-english",
        # "sat-en",
        # "aqua-rat",
        ################ 여기까지 틸다

        # "gaokao-chinese",

        #"lsat-ar",
        #"lsat-lr", "lsat-rc",
         "logiqa-en", 
        # "logiqa-zh",
        # "gaokao-mathcloze",
        # "jec-qa-kd", 
        # "jec-qa-ca",
        # "math",
        # "sat-en-without-passage",
    ]
    setting_name_list = [
        # 'few-shot',
        # 'few-shot-CoT',
        'zero-shot',
        # 'zero-shot-CoT'
    ]
    skip_stage_1 = False
    skip_stage_2 = True
    skip_stage_3 = True
    
    is_debug = False

    chat_mode = True
    work_items = []
    for dataset_name in dataset_name_list:
        for setting_name in setting_name_list:
            dataset = dataset_loader.load_dataset(
                dataset_name, setting_name, dataset_dir,
                prompt_path=raw_prompt_path, max_tokens=2048,
                end_of_example="<END>\n", verbose=True, chat_mode=chat_mode)
            
            # 랜덤 샘플링 0112
            #dataset = take_random_subset(dataset, SUBSET_N, seed=SUBSET_SEED)

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
            
            # # test용 5개만
            # dataset = dataset[:5]
            #dataset = take_random_subset(dataset, SUBSET_N, seed=SUBSET_SEED)
            
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
    
    # ===== [추가_0112] 최종 로깅 =====
    if LATENCIES:
        xs = sorted(LATENCIES)
        avg = sum(xs) / len(xs)
        logging.info(
            f"LATENCY SUMMARY | n={len(xs)} | avg={avg:.2f}s"
        )