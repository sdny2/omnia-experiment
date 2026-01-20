# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json


def read_jsonl(path):
    with open(path, encoding='utf8') as fh:
        results = []
        for line in fh:
            if line is None:
                continue
            try:
                results.append(json.loads(line) if line != "null" else line)
            except Exception as e:
                print(e)
                print(path)
                print(line)
                raise e
    return results


def save_jsonl(lines, directory):
    with open(directory, 'w', encoding='utf8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# def extract_answer(js):
#     # print("type(js):", type(js))
#     try:
#         if js is None or js == 'null':
#             # print("js is None or 'null'", js)
#             return ""
#         answer = ""
#         if isinstance(js, str):
#             answer = js
#         elif 'text' in js.choices[0]:
#             answer = js["choices"][0]["text"]
#         else:
#             answer = js.choices[0].message.content
#             # answer = js['']
#         return answer
#     except Exception as e:
#         print(e)
#         print(js)
#         return ""


def extract_answer(js):
    """
    다양한 형태의 응답에서 '텍스트 답변'을 최대한 유연하게 추출한다.

    지원 케이스:
    - None / 'null' → ""
    - str → 그대로 반환
    - dict:
        - js["answer"]               # 우리가 만든 {answer: "..."} 형식
        - js["text"]                 # 단순 text 필드
        - js["generation"]           # 일부 모델
        - js["output_text"]          # 일부 모델
        - js["content"]              # Claude / 기타
        - js["choices"][0]["..."]    # OpenAI 스타일 (하위 호환용)
        - js["raw_bedrock_response"] # Bedrock 원본에서 재파싱 (선택적)
    """
    try:
        if js is None or js == "null":
            return ""

        # 이미 문자열이면 그대로 반환
        if isinstance(js, str):
            return js

        # dict 형태일 때
        if isinstance(js, dict):
            # 1) 우리가 직접 만든 형식: {"answer": "...", "input_tokens": ..., ...}
            if isinstance(js.get("answer"), str) and js["answer"].strip():
                return js["answer"]

            # 2) 단순 text / generation / output_text 필드
            for key in ["text", "generation", "output_text"]:
                val = js.get(key)
                if isinstance(val, str) and val.strip():
                    return val

            # 3) Claude 스타일 / content 필드
            content = js.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                # Claude 3: content: [{ "type": "text", "text": "..." }, ...]
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                if texts:
                    return "".join(texts).strip()

            # 4) OpenAI 스타일 choices (하위 호환용) - js["choices"]
            choices = js.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    # chat 스타일
                    msg = first.get("message")
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        if isinstance(c, str) and c.strip():
                            return c
                    # text completion 스타일
                    txt = first.get("text")
                    if isinstance(txt, str) and txt.strip():
                        return txt

            # 5) raw_bedrock_response가 있을 때 거기서 다시 시도
            raw = js.get("raw_bedrock_response")
            if isinstance(raw, dict):
                # Claude 3 패턴
                r_content = raw.get("content")
                if isinstance(r_content, list):
                    texts = []
                    for block in r_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    if texts:
                        return "".join(texts).strip()
                # Llama / 기타: generation / outputs 등
                if isinstance(raw.get("generation"), str):
                    return raw["generation"]
                outputs = raw.get("outputs")
                if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
                    if isinstance(outputs[0].get("text"), str):
                        return outputs[0]["text"]

        # dict도 아니고 string도 아니면 그냥 문자열로 캐스팅해서 반환
        return str(js)
    except Exception as e:
        print("extract_answer error:", e)
        print("raw js:", js)
        return ""