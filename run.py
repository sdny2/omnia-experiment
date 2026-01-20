import argparse
import json
import requests


def run_model_inference(
    prompt: str,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    user_id: str,
    session_id: str,
    endpoint: str,
    timeout: int = 60,
):
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "model": {
            "model": model_name,
            "use_model_server": True
        },
        "prompt": prompt,
        "invoke_kwargs": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        }
    }

    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Run model inference via OMNIA model-inference server"
    )

    # required
    parser.add_argument(
        "--prompt",
        type=str,
        # required=True,
        default=None,
        help="Prompt text to send to the model"
    )

    # optional generation args
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=111,
        help="Maximum number of new tokens to generate (default: 111)"
    )

    # model / infra args
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://165.132.80.39:8001/model-inference",
        help="Inference server endpoint"
    )

    # routing / session
    parser.add_argument(
        "--user_id",
        type=str,
        default="demo-user",
        help="User ID (default: demo-user)"
    )
    parser.add_argument(
        "--session_id",
        type=str,
        default="working-agent-session",
        help="Session ID (default: working-agent-session)"
    )

    args = parser.parse_args()

    if args.prompt is not None:
        test_dataset = [args.prompt]
    else:
        # TODO : test dataset 불러오기
        test_dataset = [
            'what is 1 + 1?',
            'what is 1 + 2?',
            'what is 1 + 3?',
        ]

    for test_sample_idx, prompt in enumerate(test_dataset):
        result = run_model_inference(
            prompt=prompt,
            model_name=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            user_id=args.user_id,
            session_id=args.session_id,
            endpoint=args.endpoint,
        )

        # TODO : 결과물을 json 파일로 저장하기. 폴더가 없으면 생성하도록 하고 내가 사용하고 있는 config (or argument)가 구분되게 저장
        # 이 때 config는 yaml로 저장
        print('='*50)
        print(f'Test sample idx : {test_sample_idx}')
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()