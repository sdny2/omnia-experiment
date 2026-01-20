# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from src import post_process, utils, dataset_loader
from src import evaluation
import os


def _get_token_counts_from_item(item):
    """
    한 줄(item)에서 input_tokens, output_tokens를 추출하는 헬퍼.
    - {"answer": "...", "input_tokens": 10, "output_tokens": 20}
    - {"usage": {"input_tokens": 10, "output_tokens": 20}, ...}
    - {"input_token_count": 10, "output_token_count": 20, ...}
    등 여러 패턴을 최대한 커버.
    """
    in_tok = 0
    out_tok = 0

    if not isinstance(item, dict):
        return 0, 0

    # # 1) 우리가 직접 만든 {answer, input_tokens, output_tokens} 형태
    # if "input_tokens" in item or "output_tokens" in item:
    #     if isinstance(item.get("input_tokens"), (int, float)):
    #         in_tok = int(item["input_tokens"])
    #     if isinstance(item.get("output_tokens"), (int, float)):
    #         out_tok = int(item["output_tokens"])
    #     return in_tok, out_tok

    # # 2) usage 딕셔너리 안에 있는 형태
    # usage = item.get("usage", {})
    # if isinstance(usage, dict):
    #     if isinstance(usage.get("input_tokens"), (int, float)):
    #         in_tok = int(usage["input_tokens"])
    #     if isinstance(usage.get("output_tokens"), (int, float)):
    #         out_tok = int(usage["output_tokens"])
    #     if in_tok or out_tok:
    #         return in_tok, out_tok

    # # 3) 다른 이름을 쓰는 경우 (예: input_token_count / output_token_count)
    # if isinstance(item.get("input_token_count"), (int, float)):
    #     in_tok = int(item["input_token_count"])
    # if isinstance(item.get("output_token_count"), (int, float)):
    #     out_tok = int(item["output_token_count"])
    
    in_tok = item.get("input_tokens", 0)
    out_tok = item.get("output_tokens", 0)
    return in_tok, out_tok


def compute_total_token_usage_for_outputs(output_dir, dataset_name_list, setting_name_list, gpt_model):
    """
    outputs/predict.{gpt_model}.{dataset_name}.{setting_name}.jsonl 파일들을 모두 읽어서
    전체 input_tokens, output_tokens 합계를 계산한다.

    반환값: (total_input_tokens, total_output_tokens)
    """
    total_input_tokens = 0
    total_output_tokens = 0

    for dataset_name in dataset_name_list:
        for setting_name in setting_name_list:
            output_path = os.path.join(
                output_dir,
                "outputs",
                f"predict.{gpt_model}.{dataset_name}.{setting_name}.jsonl"
            )
            if not os.path.exists(output_path):
                # 결과 없는 세팅은 스킵
                continue

            try:
                items = utils.read_jsonl(output_path)
            except Exception as e:
                print(f"[TOKEN USAGE] Failed to read {output_path}: {e}")
                continue

            file_input = 0
            file_output = 0
            for item in items:
                in_tok, out_tok = _get_token_counts_from_item(item)
                file_input += in_tok
                file_output += out_tok

            total_input_tokens += file_input
            total_output_tokens += file_output

    #         print(f"[TOKEN USAGE] {dataset_name} / {setting_name}: "
    #               f"input={file_input}, output={file_output}")

    # print(f"[TOKEN USAGE] TOTAL input_tokens={total_input_tokens}=${0.00072*total_input_tokens/1000}, "
    #       f"TOTAL output_tokens={total_output_tokens}=${0.00072*total_output_tokens/1000}")

    return total_input_tokens, total_output_tokens


if __name__ == "__main__":
    dataset_dir = "data/v1_1"
    raw_prompt_path = "./data/few_shot_prompts.csv"
    # gpt_model = 'gpt-35-turbo'
    # gpt_model = 'llama70b-test'
    output_dir = "./outputs/{}".format('llama70b')
    gpt_model = 'aws'
    chat_mode = True

    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    dataset_name_list = ["sat-math", "gaokao-geography"]
    dataset_name_list = [
        "aqua-rat",
        "sat-math", 
        "sat-en",
        "gaokao-english",
        "gaokao-geography", 
        "gaokao-history",
        "gaokao-biology", 
        "gaokao-chemistry", 
        "gaokao-physics",
        "gaokao-mathqa",

        "logiqa-en", 
        "logiqa-zh",
        "gaokao-mathcloze",
        "sat-en-without-passage",
        "gaokao-chinese",
        "math",
        "jec-qa-kd", 
        "jec-qa-ca",
        "lsat-ar", 
        "lsat-lr", 
        "lsat-rc",        
    ]

    english_qa_dataset_name_list = [
        "sat-math",
        "aqua-rat",
        "logiqa-en",
        "lsat-ar", 
        "lsat-lr", 
        "lsat-rc",
        "sat-math", 
        "sat-en",
        "sat-en-without-passage",
    ]

    chinese_qa_dataset_name_list = [
        "logiqa-zh",
        "jec-qa-kd",
         "jec-qa-ca",
        "gaokao-chinese",
        "gaokao-english",
        "gaokao-geography", 
        "gaokao-history",
        "gaokao-biology", 
        "gaokao-chemistry",
        "gaokao-physics",
        "gaokao-mathqa",
    ]

    setting_name_list = [
        'zero-shot',
        # 'zero-shot-CoT',
        # 'few-shot',
        # 'few-shot-CoT',
    ]

    sum_list = [0] * len(setting_name_list)
    english_qa_model_results = {}
    chinese_qa_model_results = {}
    inp_toks_intotal, out_toks_intotal = 0, 0
    print("\t" + "\t".join(setting_name_list))
    for dataset_name in dataset_name_list:
        model_results = {}
        accuracy_list = []
        for setting_id, setting_name in enumerate(setting_name_list):
            dataset = dataset_loader.load_dataset(
                dataset_name, setting_name, dataset_dir,
                prompt_path=raw_prompt_path, max_tokens=2048,
                end_of_example="<END>\n", chat_mode=chat_mode)
            utils.save_jsonl(dataset, os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.jsonl"))
            output_path = os.path.join(
                output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.jsonl')
            first_stage_output_path = os.path.join(
                output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.first_stage.jsonl')
            second_stage_input_path = os.path.join(
                output_dir, "inputs", f"{dataset_name}.{setting_name}.second_stage.jsonl")

            if not os.path.exists(output_path):
                print("dataset {0} setting {1} doesn't have results".format(dataset_name, setting_name))
                print(f'See {output_path}')
                accuracy_list.append("0")
                continue

            context_list = [item['context'] for item in dataset]

            result_for_human = dataset_loader.load_dataset_as_result_schema(
                dataset_name, dataset_dir
            )

            output_jsons = utils.read_jsonl(output_path)
            if 'zero-shot' in setting_name:
                first_stage_output_jsons = utils.read_jsonl(first_stage_output_path)
                second_stage_input_jsons = utils.read_jsonl(second_stage_input_path)

            for i in range(len(result_for_human)):
                result_for_human[i].model_input = dataset[i]["context"]
                result_for_human[i].model_output = utils.extract_answer(output_jsons[i])
                result_for_human[i].parse_result = post_process.post_process(dataset_name, setting_name,
                                                                             result_for_human[i].model_output)
                result_for_human[i].is_correct = evaluation.evaluate_single_sample(
                    dataset_name, result_for_human[i].parse_result, result_for_human[i].label)

                if 'zero-shot' in setting_name:
                    result_for_human[i].first_stage_output = utils.extract_answer(first_stage_output_jsons[i])
                    result_for_human[i].second_stage_input = second_stage_input_jsons[i]["context"]

            if 'few-shot' in setting_name:
                correct_format = 0
                for i in range(len(result_for_human)):
                    if post_process.try_parse_few_shot_pattern(
                            result_for_human[i].model_output, dataset_name, setting_name):
                        correct_format += 1
                correct_ratio = correct_format / len(result_for_human)
            correct_numer = len([item for item in result_for_human if item.is_correct])
            accuracy = correct_numer / len(result_for_human)
            accuracy_list.append("{0:.2%}".format(accuracy))
            sum_list[setting_id] += accuracy

            # 전체 토큰 사용량 계산
            inp_toks, out_toks = compute_total_token_usage_for_outputs(
                output_dir=output_dir,
                dataset_name_list=dataset_name_list,
                setting_name_list=setting_name_list,
                gpt_model=gpt_model,
            )
            inp_toks_intotal += inp_toks
            out_toks_intotal += out_toks
        print("\t".join([dataset_name] + accuracy_list))
        model_results[dataset_name] = accuracy_list[0]
        if dataset_name in english_qa_dataset_name_list:
            english_qa_model_results[dataset_name] = accuracy_list
        elif dataset_name in chinese_qa_dataset_name_list:
            chinese_qa_model_results[dataset_name] = accuracy_list

    print(f'Cost : \n\tInput=${0.00072*inp_toks_intotal/1000}\n\tOutput=${0.00072*out_toks_intotal/1000}')
    average_list = []
    for item in sum_list:
        average_list.append("{0:.2%}".format(item / len(dataset_name_list)))
    print("\t".join(["average for all datasets"] + average_list))

    # average accuracy for English QA datasets
    sum_list = [0] * len(setting_name_list)
    for dataset_name in english_qa_dataset_name_list:
        for setting_id, setting_name in enumerate(setting_name_list):
            if setting_name in ['zero-shot', 'zero-shot-CoT', 'few-shot', 'few-shot-CoT']:
                sum_list[setting_id] += float(english_qa_model_results[dataset_name][setting_id][:-1])
    average_list = []
    for item in sum_list:
        average_list.append("{0:.2%}".format(item / len(english_qa_dataset_name_list) * 0.01))
    print("\t".join(["average for English multi choice QA"] + average_list))

    # average accuracy for Chinese QA datasets
    sum_list = [0] * len(setting_name_list)
    for dataset_name in chinese_qa_dataset_name_list:
        for setting_id, setting_name in enumerate(setting_name_list):
            if setting_name in ['zero-shot', 'zero-shot-CoT', 'few-shot', 'few-shot-CoT']:
                sum_list[setting_id] += float(chinese_qa_model_results[dataset_name][setting_id][:-1])
    average_list = []
    for item in sum_list:
        average_list.append("{0:.2%}".format(item / len(chinese_qa_dataset_name_list) * 0.01))
    print("\t".join(["average for Chinese multi choice QA"] + average_list))